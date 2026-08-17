## Supplementary checkout material

The following excerpts are from the same Lantern checkout. They are included as additional
repository context for the coding-agent task; they are not a second task. Some snippets are
from adjacent packages that were not copied into the original incident bundle. Preserve the
existing architectural constraints when reasoning about them: PostgreSQL is the durable
ordering source, Redis is a delivery mechanism, and tenant scope is part of every lookup.

## `apps/api/src/infra/postgres/idempotency-repository.ts`

```ts
import type { PoolClient } from "pg";
import type { Job } from "../../domain/job.js";

export type StoredIdempotency = {
  tenantId: string;
  key: string;
  requestHash: string;
  response: Job;
  createdAt: Date;
};

export class IdempotencyRepository {
  async find(client: PoolClient, tenantId: string, key: string): Promise<StoredIdempotency | null> {
    const result = await client.query<StoredIdempotency>(
      `SELECT tenant_id AS "tenantId", key, request_hash AS "requestHash",
              response_json AS response, created_at AS "createdAt"
         FROM idempotency_keys
        WHERE tenant_id = $1 AND key = $2`,
      [tenantId, key],
    );
    return result.rows[0] ?? null;
  }

  async insert(
    client: PoolClient,
    value: Omit<StoredIdempotency, "createdAt">,
  ): Promise<void> {
    await client.query(
      `INSERT INTO idempotency_keys
         (tenant_id, key, request_hash, response_json)
       VALUES ($1, $2, $3, $4)`,
      [value.tenantId, value.key, value.requestHash, JSON.stringify(value.response)],
    );
  }
}
```

The idempotency key is supplied by the client and is only unique within a tenant. A retry with
the same key but a different canonical request hash is a conflict. A retry with the same hash
must return the stored job representation, including the original id and timestamps. The
repository intentionally leaves the unique-violation retry path to the application service so
that the transaction can re-read the row after PostgreSQL resolves the concurrent insert.

## `apps/api/src/infra/postgres/transaction.ts`

```ts
import type { Pool, PoolClient } from "pg";

export async function inTransaction<T>(pool: Pool, fn: (client: PoolClient) => Promise<T>): Promise<T> {
  const client = await pool.connect();
  try {
    await client.query("BEGIN");
    const value = await fn(client);
    await client.query("COMMIT");
    return value;
  } catch (error) {
    try { await client.query("ROLLBACK"); } catch { /* preserve original error */ }
    throw error;
  } finally {
    client.release();
  }
}

export async function serializable<T>(pool: Pool, fn: (client: PoolClient) => Promise<T>): Promise<T> {
  return inTransaction(pool, async (client) => {
    await client.query("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE");
    return fn(client);
  });
}
```

The helper is used for operations that allocate an idempotency record or a next event
sequence. It does not make Redis operations part of the database transaction. Redis enqueue is
performed after commit and is safe to repeat because the outbox publisher records its delivery
cursor and event ids.

## `apps/api/src/application/job-command-service.ts`

```ts
import { createHash } from "node:crypto";
import type { Pool, PoolClient } from "pg";
import { inTransaction } from "../infra/postgres/transaction.js";
import { IdempotencyRepository } from "../infra/postgres/idempotency-repository.js";
import { JobRepository } from "../infra/postgres/job-repository.js";
import { OutboxRepository } from "../infra/postgres/outbox-repository.js";

type CreateInput = { tenantId: string; idempotencyKey: string; width: number; height: number; assetId: string };

function requestHash(input: CreateInput): string {
  const canonical = JSON.stringify({ assetId: input.assetId, height: input.height, width: input.width });
  return createHash("sha256").update(canonical).digest("hex");
}

export async function createJob(
  pool: Pool,
  jobs: JobRepository,
  idempotency: IdempotencyRepository,
  outbox: OutboxRepository,
  input: CreateInput,
) {
  const hash = requestHash(input);
  return inTransaction(pool, async (client: PoolClient) => {
    const existing = await idempotency.find(client, input.tenantId, input.idempotencyKey);
    if (existing) {
      if (existing.requestHash !== hash) throw new Error("idempotency key reused with different request");
      return existing.response;
    }
    const job = await jobs.insertQueued(client, input.tenantId, input);
    await outbox.append(client, jobs.createdEvent(job));
    try {
      await idempotency.insert(client, {
        tenantId: input.tenantId,
        key: input.idempotencyKey,
        requestHash: hash,
        response: job,
      });
    } catch (error) {
      // A concurrent transaction can win the unique key. The caller must retry the
      // transaction and then return the winner after comparing requestHash.
      throw error;
    }
    return job;
  });
}
```

## `apps/api/src/infra/postgres/job-repository-locking.ts`

```ts
import type { PoolClient } from "pg";
import type { Job, JobState } from "../../domain/job.js";

export class LockedJobRepository {
  async lock(client: PoolClient, tenantId: string, jobId: string): Promise<Job | null> {
    const row = await client.query<Job>(
      `SELECT id, tenant_id AS "tenantId", state, version, next_event_sequence AS "nextEventSequence",
              lease_id AS "leaseId", updated_at AS "updatedAt"
         FROM jobs
        WHERE tenant_id = $1 AND id = $2
        FOR UPDATE`,
      [tenantId, jobId],
    );
    return row.rows[0] ?? null;
  }

  async updateState(client: PoolClient, job: Job, nextState: JobState, leaseId: string | null): Promise<Job> {
    const next = {
      ...job,
      state: nextState,
      version: job.version + 1,
      nextEventSequence: job.nextEventSequence + 1,
      leaseId,
      updatedAt: new Date(),
    };
    await client.query(
      `UPDATE jobs SET state = $1, version = $2, next_event_sequence = $3,
                       lease_id = $4, updated_at = $5
        WHERE tenant_id = $6 AND id = $7`,
      [next.state, next.version, next.nextEventSequence, next.leaseId,
       next.updatedAt, next.tenantId, next.id],
    );
    return next;
  }
}
```

Every state-changing command locks the job row before reading its current version. A command
must create the event payload from the returned row, append the event, append the outbox row,
and commit all three changes together. A worker completion is not allowed to use a job object
that it read before a lease check.

## `apps/worker/src/lease-store.ts`

```ts
import type { RedisClientType } from "redis";

const acquireScript = `
  if redis.call('exists', KEYS[1]) == 0 then
    redis.call('set', KEYS[1], ARGV[1], 'PX', ARGV[2]); return 1
  end
  return 0
`;

const renewScript = `
  if redis.call('get', KEYS[1]) == ARGV[1] then
    redis.call('pexpire', KEYS[1], ARGV[2]); return 1
  end
  return 0
`;

const releaseScript = `
  if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
  end
  return 0
`;

export class LeaseStore {
  constructor(private readonly redis: RedisClientType) {}

  async acquire(key: string, owner: string, ttlMs: number): Promise<boolean> {
    return Number(await this.redis.eval(acquireScript, { keys: [key], arguments: [owner, String(ttlMs)] })) === 1;
  }

  async renew(key: string, owner: string, ttlMs: number): Promise<boolean> {
    return Number(await this.redis.eval(renewScript, { keys: [key], arguments: [owner, String(ttlMs)] })) === 1;
  }

  async release(key: string, owner: string): Promise<boolean> {
    return Number(await this.redis.eval(releaseScript, { keys: [key], arguments: [owner] })) === 1;
  }
}
```

The Redis lease is useful for work ownership, but it is not itself the durable event authority.
The completion transaction still needs a compare-and-set against the lease id stored on the
job row, or an equivalent authoritative lease table. A successful Redis renewal and a stale
PostgreSQL row must never be interpreted as permission to publish a completion.

## `apps/worker/src/lease-renewer.ts`

```ts
export class LeaseRenewer {
  private timer: NodeJS.Timeout | undefined;
  private lost = false;

  constructor(
    private readonly renew: () => Promise<boolean>,
    private readonly everyMs: number,
  ) {}

  start(): void {
    this.timer = setInterval(async () => {
      try {
        if (!(await this.renew())) this.lost = true;
      } catch {
        this.lost = true;
      }
    }, this.everyMs);
  }

  hasLost(): boolean { return this.lost; }

  stop(): void {
    if (this.timer) clearInterval(this.timer);
    this.timer = undefined;
  }
}
```

The renewer can report a lost lease while a render is still running. Rendering may finish for
resource accounting, but the worker must discard the result unless the completion transaction
revalidates ownership. The boolean is advisory and can race with expiry; it is not a substitute
for a conditional update under a database lock.

## `apps/worker/src/render/render-engine.ts`

```ts
export type RenderResult = {
  bytes: Uint8Array;
  contentType: string;
  checksum: string;
  durationMs: number;
};

export interface RenderEngine {
  render(input: { assetId: string; width: number; height: number; signal: AbortSignal }): Promise<RenderResult>;
}

export async function renderWithLease(
  engine: RenderEngine,
  input: { assetId: string; width: number; height: number },
  signal: AbortSignal,
): Promise<RenderResult> {
  const started = performance.now();
  const result = await engine.render({ ...input, signal });
  return { ...result, durationMs: performance.now() - started };
}
```

The render output is immutable content-addressed data. Uploading it before the completion
transaction is acceptable if orphan cleanup is idempotent. The worker must not make the object
visible as the job's result, or emit a success event, until it owns the job at commit time.

## `apps/worker/src/job-runner.ts`

```ts
import { randomUUID } from "node:crypto";
import type { Pool } from "pg";
import { inTransaction } from "../../api/src/infra/postgres/transaction.js";
import { LockedJobRepository } from "../../api/src/infra/postgres/job-repository-locking.js";
import { EventStore } from "../../packages/events/src/event-store.js";
import { OutboxRepository } from "../../api/src/infra/postgres/outbox-repository.js";

export async function markRunning(pool: Pool, jobs: LockedJobRepository, events: EventStore, outbox: OutboxRepository, tenantId: string, jobId: string, leaseId: string) {
  return inTransaction(pool, async (client) => {
    const current = await jobs.lock(client, tenantId, jobId);
    if (!current || current.leaseId !== leaseId) return { ok: false as const, reason: "lost-lease" };
    const next = await jobs.updateState(client, current, "running", leaseId);
    const event = { eventId: randomUUID(), tenantId, jobId, sequence: next.version, state: next.state, type: "job.running" as const, occurredAt: next.updatedAt.toISOString(), payload: {} };
    await events.append(client, event);
    await outbox.append(client, event);
    return { ok: true as const, job: next };
  });
}

export async function complete(pool: Pool, jobs: LockedJobRepository, events: EventStore, outbox: OutboxRepository, tenantId: string, jobId: string, leaseId: string, output: { objectKey: string; checksum: string }) {
  return inTransaction(pool, async (client) => {
    const current = await jobs.lock(client, tenantId, jobId);
    if (!current || current.leaseId !== leaseId || current.state !== "running") {
      return { ok: false as const, reason: "lost-lease" };
    }
    const next = await jobs.updateState(client, current, "succeeded", null);
    const event = { eventId: randomUUID(), tenantId, jobId, sequence: next.version, state: next.state, type: "job.succeeded" as const, occurredAt: next.updatedAt.toISOString(), payload: output };
    await events.append(client, event);
    await outbox.append(client, event);
    return { ok: true as const, job: next };
  });
}
```

The important property in both methods is that the row is locked and the lease is checked in
the same transaction that changes state and appends the event. A zero-row conditional update or
a failed lease comparison is a normal lost-work result. It must not be logged as a successful
completion or retried with the stale version.

## `apps/worker/src/outbox-publisher.ts`

```ts
import type { RedisClientType } from "redis";
import type { Pool } from "pg";
import { OutboxRepository } from "../../api/src/infra/postgres/outbox-repository.js";

export class OutboxPublisher {
  constructor(
    private readonly pool: Pool,
    private readonly outbox: OutboxRepository,
    private readonly redis: RedisClientType,
  ) {}

  async publishBatch(limit: number): Promise<number> {
    const rows = await this.outbox.unpublished(this.pool, limit);
    let count = 0;
    for (const row of rows) {
      await this.redis.xAdd(`tenant:${row.tenantId}:jobs`, "*", {
        eventId: row.eventId,
        jobId: row.jobId,
        sequence: String(row.sequence),
        payload: JSON.stringify(row.payload),
      });
      await this.outbox.markPublished(this.pool, row.id);
      count += 1;
    }
    return count;
  }
}
```

The publisher is at-least-once. It can crash after `XADD` and before `markPublished`, so the
consumer must deduplicate by `(tenantId, jobId, sequence)` or event id. Marking a row published
before Redis acknowledges it would create a permanent gap. The durable outbox row remains the
recovery source.

## `packages/events/src/event-schema.ts`

```ts
import { z } from "zod";

export const jobEventSchema = z.object({
  eventId: z.string().uuid(),
  tenantId: z.string().min(1),
  jobId: z.string().min(1),
  sequence: z.number().int().positive(),
  occurredAt: z.string().datetime(),
  type: z.enum(["job.queued", "job.running", "job.succeeded", "job.failed", "job.cancelled"]),
  state: z.enum(["queued", "running", "succeeded", "failed", "cancelled"]),
  payload: z.record(z.string(), z.unknown()),
});

export type ValidatedJobEvent = z.infer<typeof jobEventSchema>;

export function eventKey(event: Pick<ValidatedJobEvent, "tenantId" | "jobId" | "sequence">): string {
  return `${event.tenantId}:${event.jobId}:${event.sequence}`;
}
```

Event validation must happen before a message is handed to a reducer. The tenant id in the
payload is not trusted as an authorization grant; the stream and the authenticated request
still determine which tenant may receive it.

## `packages/events/src/projector.ts`

```ts
import type { PoolClient } from "pg";
import type { JobEvent } from "./job-events.js";

export async function applyEvent(client: PoolClient, event: JobEvent): Promise<"applied" | "stale" | "gap"> {
  const current = await client.query<{ state: string; version: number }>(
    `SELECT state, version FROM jobs WHERE tenant_id = $1 AND id = $2 FOR UPDATE`,
    [event.tenantId, event.jobId],
  );
  const job = current.rows[0];
  if (!job) return "stale";
  if (event.sequence <= job.version) return "stale";
  if (event.sequence !== job.version + 1) return "gap";
  await client.query(
    `UPDATE jobs SET state = $1, version = $2, updated_at = $3
      WHERE tenant_id = $4 AND id = $5 AND version = $6`,
    [event.state, event.sequence, event.occurredAt, event.tenantId, event.jobId, job.version],
  );
  return "applied";
}
```

The projection should either apply a contiguous event under a row lock or retain the event for
gap repair. It must not apply a lower sequence after a terminal state, and it must not silently
advance over a missing event. If the write side already updates the projection transactionally,
the consumer applies only recovery events and must use the same monotonic checks.

## `packages/events/src/replay-cursor.ts`

```ts
export type Cursor = { jobId: string; sequence: number };

export function parseCursor(value: string | undefined): Cursor | null {
  if (!value) return null;
  const separator = value.lastIndexOf(":");
  if (separator <= 0) return null;
  const jobId = value.slice(0, separator);
  const sequence = Number(value.slice(separator + 1));
  return Number.isSafeInteger(sequence) && sequence > 0 ? { jobId, sequence } : null;
}

export function cursorFor(event: { jobId: string; sequence: number }): string {
  return `${event.jobId}:${event.sequence}`;
}
```

The cursor is a replay position, not an authorization selector. The authenticated tenant and
job access check must be performed before using it in a query. A malformed cursor is treated as
no cursor or a client error according to the API contract; it must never widen the query.

## `apps/api/src/routes/events-stream.ts`

```ts
export async function eventsStream(request: Request, response: Response, deps: Dependencies) {
  const tenantId = request.auth.tenantId;
  const jobId = request.params.jobId;
  await deps.authorizeJob(tenantId, jobId);

  const live = deps.eventBus.subscribe({ tenantId, jobId }, (event) => {
    if (event.sequence > lastSequence) {
      lastSequence = event.sequence;
      response.write(`id: ${event.jobId}:${event.sequence}\ndata: ${JSON.stringify(event)}\n\n`);
    }
  });
  try {
    const watermark = await deps.events.maxSequence(tenantId, jobId);
    let lastSequence = Number(deps.parseLastEventId(request.headers.get("last-event-id"))) || 0;
    const replay = await deps.events.after(tenantId, jobId, lastSequence, watermark);
    for (const event of replay) {
      if (event.sequence > lastSequence) {
        lastSequence = event.sequence;
        response.write(`id: ${event.jobId}:${event.sequence}\ndata: ${JSON.stringify(event)}\n\n`);
      }
    }
    // Read once more after subscription to cover the watermark boundary.
    const catchUp = await deps.events.after(tenantId, jobId, lastSequence, undefined);
    for (const event of catchUp) {
      if (event.sequence > lastSequence) {
        lastSequence = event.sequence;
        response.write(`id: ${event.jobId}:${event.sequence}\ndata: ${JSON.stringify(event)}\n\n`);
      }
    }
    await deps.waitForDisconnect(request);
  } finally {
    live();
  }
}
```

This sketch intentionally shows the subscribe-before-replay shape and a second durable read.
The actual route should use a small helper to avoid the variable-initialization ordering shown
in this excerpt. Live events and replay both pass through the same monotonic cursor, so a
duplicate caused by the boundary race is invisible to the client.

## `apps/web/src/dashboard/job-reducer.ts`

```ts
import type { JobEvent } from "../../../packages/events/src/job-events.js";

const terminal = new Set(["succeeded", "failed", "cancelled"]);

export function reduceJob(state: JobView, event: JobEvent): JobView {
  if (event.tenantId !== state.tenantId || event.jobId !== state.id) return state;
  if (event.sequence <= state.sequence) return state;
  if (terminal.has(state.state)) return state;
  if (event.sequence !== state.sequence + 1) {
    return { ...state, needsResync: true };
  }
  return {
    ...state,
    state: event.state,
    sequence: event.sequence,
    result: event.state === "succeeded" ? event.payload : state.result,
    needsResync: false,
  };
}
```

The browser reducer is a final defense, not the durable consistency mechanism. A terminal view
must not be reopened by a delayed `running` event. A gap should trigger a resync rather than be
silently accepted as a newer state.

## `apps/api/src/routes/jobs.test.ts`

```ts
describe("idempotent create", () => {
  it("returns one durable job for concurrent identical requests", async () => {
    const [a, b] = await Promise.all([
      create({ tenantId: "tenant-a", idempotencyKey: "request-7", width: 800, height: 600, assetId: "asset-1" }),
      create({ tenantId: "tenant-a", idempotencyKey: "request-7", width: 800, height: 600, assetId: "asset-1" }),
    ]);
    expect(a.job.id).toBe(b.job.id);
    expect(await countJobs("tenant-a")).toBe(1);
    expect(await countIdempotencyRows("tenant-a", "request-7")).toBe(1);
  });

  it("rejects a changed request under an existing key", async () => {
    await create({ tenantId: "tenant-a", idempotencyKey: "request-8", width: 800, height: 600, assetId: "asset-1" });
    await expect(create({ tenantId: "tenant-a", idempotencyKey: "request-8", width: 801, height: 600, assetId: "asset-1" })).rejects.toThrow(/different request/);
  });
});
```

The concurrency test must use a real unique constraint or a faithful transaction test double.
Two independent reads followed by two inserts are not an adequate test of the production
behavior. The response comparison should include all fields that the client persists.

## `apps/worker/src/job-runner.test.ts`

```ts
describe("completion ownership", () => {
  it("does not emit success after another worker takes the lease", async () => {
    const fixture = await makeWorkerFixture();
    await fixture.queue("tenant-a", "job-1");
    const first = await fixture.claim("worker-a");
    await fixture.expireLease(first.leaseId);
    const second = await fixture.claim("worker-b");
    const staleResult = await fixture.finish(first, { objectKey: "orphan/a", checksum: "a" });
    expect(staleResult.ok).toBe(false);
    expect(await fixture.eventsFor("job-1")).not.toContainEqual(expect.objectContaining({ state: "succeeded", payload: { objectKey: "orphan/a", checksum: "a" } }));
    await fixture.finish(second, { objectKey: "result/b", checksum: "b" });
    expect(await fixture.terminalSequences("job-1")).toEqual([2]);
  });
});
```

The test should force the lease handoff between rendering and completion. A unit test that only
sets a boolean on the first worker misses the race between Redis expiry and the database write.

## `apps/api/src/routes/events-stream.test.ts`

```ts
it("does not duplicate an event crossing replay and subscription", async () => {
  const fixture = await makeStreamFixture({ durableThrough: 12 });
  const responsePromise = fixture.connect({ tenantId: "tenant-a", jobId: "job-1", lastEventId: "job-1:11" });
  await fixture.publish({ tenantId: "tenant-a", jobId: "job-1", sequence: 12 });
  const response = await responsePromise;
  expect(response.events.map((event) => event.sequence)).toEqual([12]);
});

it("never reads another tenant when a cursor names the same job id", async () => {
  const fixture = await makeStreamFixture({ durableThrough: 4, otherTenantThrough: 9 });
  const response = await fixture.connect({ tenantId: "tenant-a", jobId: "same-id", lastEventId: "same-id:8" });
  expect(response.events.every((event) => event.tenantId === "tenant-a")).toBe(true);
});
```

The first test controls the timing so the event is visible in either the replay query or the
live callback, but never twice. The second test is a security regression test as well as an
ordering test: `(tenant_id, job_id)` must be present in every durable query.

## `db/migrations/006_consistency_constraints.sql`

```sql
ALTER TABLE idempotency_keys
  ADD CONSTRAINT idempotency_keys_tenant_key_unique UNIQUE (tenant_id, key);

CREATE INDEX job_events_tenant_job_sequence
  ON job_events (tenant_id, job_id, sequence);

CREATE UNIQUE INDEX job_events_tenant_job_sequence_unique
  ON job_events (tenant_id, job_id, sequence);

ALTER TABLE jobs
  ADD CONSTRAINT jobs_version_nonnegative CHECK (version >= 0),
  ADD CONSTRAINT jobs_next_event_after_version CHECK (next_event_sequence > version);
```

The unique event index catches duplicate sequence allocation, but it does not allocate the next
number. Allocation still belongs inside the locked job transaction. If an online migration is
needed, create the indexes concurrently and validate constraints in a later deploy.

## `db/queries/replay.sql`

```sql
-- The application has already authorized the tenant/job pair.
SELECT event_id, tenant_id, job_id, sequence, occurred_at, type, state, payload
  FROM job_events
 WHERE tenant_id = $1
   AND job_id = $2
   AND sequence > $3
   AND ($4::bigint IS NULL OR sequence <= $4)
 ORDER BY sequence ASC;
```

Never replace the tenant predicate with a lookup by job id alone. A cursor received from a
browser is untrusted input. The upper watermark is an optimization; the second catch-up read
after subscribing is what closes the replay/live race.

## `packages/shared/src/request-hash.ts`

```ts
import { createHash } from "node:crypto";

export function stableRequestHash(value: unknown): string {
  return createHash("sha256").update(stableJson(value)).digest("hex");
}

function stableJson(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(stableJson).join(",")}]`;
  const entries = Object.entries(value as Record<string, unknown>).sort(([a], [b]) => a.localeCompare(b));
  return `{${entries.map(([key, item]) => `${JSON.stringify(key)}:${stableJson(item)}`).join(",`)}}`;
}
```

The hash must cover every field that changes the created job, including output dimensions and
render options. Trace ids, authorization headers, and transport retries are not part of the
idempotent operation. Canonicalization must be shared by all API instances.

## `packages/shared/src/result.ts`

```ts
export type Result<T, E> = { ok: true; value: T } | { ok: false; error: E };

export type DomainError =
  | { kind: "not-found" }
  | { kind: "conflict"; message: string }
  | { kind: "lost-lease" }
  | { kind: "event-gap"; expected: number; received: number };

export function isOk<T, E>(result: Result<T, E>): result is { ok: true; value: T } {
  return result.ok;
}
```

A lost lease is not an infrastructure retry by default. Retrying the stale completion can emit
an event from the wrong owner. The worker may enqueue the job again through a separate recovery
policy, but that policy must acquire a fresh lease and re-read the job.

## `apps/api/src/observability/job-metrics.ts`

```ts
export interface JobMetrics {
  commandStarted(name: string, tenantId: string): void;
  commandFinished(name: string, outcome: "ok" | "conflict" | "lost-lease" | "error", elapsedMs: number): void;
  eventGap(tenantId: string, jobId: string, expected: number, received: number): void;
  outboxRedelivery(tenantId: string, jobId: string, sequence: number): void;
}

export function redactJobId(jobId: string): string {
  return jobId.length <= 8 ? "redacted" : `${jobId.slice(0, 4)}…${jobId.slice(-4)}`;
}
```

Metrics should distinguish a duplicate delivery from an event gap and a lost lease. A spike in
redeliveries is expected during a publisher restart; a spike in gaps indicates a durable or
consumer ordering problem. Do not use a low free-memory metric from an unrelated host as a
reason to abort a worker operation in this service.

## `apps/api/src/application/retry-policy.ts`

```ts
export async function withSerializationRetry<T>(operation: () => Promise<T>, maxAttempts = 3): Promise<T> {
  let lastError: unknown;
  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    try {
      return await operation();
    } catch (error) {
      lastError = error;
      if (!isSerializationFailure(error) || attempt === maxAttempts) throw error;
      await new Promise((resolve) => setTimeout(resolve, attempt * 10));
    }
  }
  throw lastError;
}

function isSerializationFailure(error: unknown): boolean {
  return typeof error === "object" && error !== null && "code" in error && (error as { code?: string }).code === "40001";
}
```

Serialization retries are safe only when the retried operation is idempotent and the recovery
path re-reads the durable idempotency row. A retry that blindly calls `insertQueued` again can
create a second job after the first transaction committed but its response was lost.

## `apps/api/src/health/consistency-check.ts`

```ts
export async function checkJobConsistency(client: PoolClient, tenantId: string, jobId: string) {
  const job = await client.query<{ version: number; state: string }>(
    `SELECT version, state FROM jobs WHERE tenant_id = $1 AND id = $2`, [tenantId, jobId],
  );
  const events = await client.query<{ sequence: number; state: string }>(
    `SELECT sequence, state FROM job_events WHERE tenant_id = $1 AND job_id = $2 ORDER BY sequence`,
    [tenantId, jobId],
  );
  const rows = events.rows;
  const contiguous = rows.every((row, index) => row.sequence === index + 1);
  return {
    exists: job.rowCount === 1,
    eventCount: rows.length,
    eventVersion: rows.at(-1)?.sequence ?? 0,
    projectionVersion: job.rows[0]?.version ?? null,
    contiguous,
    terminal: ["succeeded", "failed", "cancelled"].includes(job.rows[0]?.state ?? ""),
  };
}
```

This check is for diagnostics and repair tooling. It must retain tenant scope and should report
inconsistency rather than mutating production state from a health endpoint. Repair jobs should
append a clearly identified event under the same sequence-allocation transaction.

## `apps/web/src/dashboard/event-client.ts`

```ts
export class EventClient {
  private source: EventSource | undefined;
  private lastEventId: string | undefined;

  connect(url: string, onEvent: (event: JobEvent) => void): void {
    const query = this.lastEventId ? `?after=${encodeURIComponent(this.lastEventId)}` : "";
    this.source = new EventSource(`${url}${query}`);
    this.source.onmessage = (message) => {
      const event = JSON.parse(message.data) as JobEvent;
      this.lastEventId = message.lastEventId;
      onEvent(event);
    };
    this.source.onerror = () => {
      this.source?.close();
      setTimeout(() => this.connect(url, onEvent), 250);
    };
  }

  close(): void {
    this.source?.close();
    this.source = undefined;
  }
}
```

The browser can reconnect after receiving an event but before persisting its cursor. The server
must make replay idempotent. The client should persist the last accepted sequence along with the
tenant/job identity and discard an event whose identity does not match the current view.

## `apps/api/src/routes/jobs.contract.ts`

```ts
export type CreateJobResponse = {
  id: string;
  tenantId: string;
  state: "queued";
  version: 1;
  nextEventSequence: 2;
  createdAt: string;
  updatedAt: string;
};

export function responseFor(job: Job): CreateJobResponse {
  return {
    id: job.id,
    tenantId: job.tenantId,
    state: "queued",
    version: job.version,
    nextEventSequence: job.nextEventSequence,
    createdAt: job.createdAt.toISOString(),
    updatedAt: job.updatedAt.toISOString(),
  };
}
```

The stored idempotency response should be the exact serialized API response or a representation
that is provably equivalent. Reconstructing a response on retry from a later job row can change
the version or state and violates the client contract.

## `docs/operations/outbox-recovery.md`

```text
1. Stop the publisher deployment, not the API or worker deployments.
2. Inspect unpublished rows by tenant and age; do not delete them as a first response.
3. Verify the event exists in job_events before manually replaying a stream message.
4. Re-run the publisher with the same event id and allow consumers to deduplicate it.
5. If a gap remains, compare the durable sequence list with the projection version.
6. Repair only through the audited sequence-allocation command.
```

Outbox recovery must preserve event ordering and tenant isolation. An operator tool that accepts
only a job id is unsafe because job ids are not necessarily globally unique across all storage
boundaries.

## `docs/architecture/state-machine.md`

```text
queued    -> running, cancelled
running   -> succeeded, failed, cancelled
succeeded -> (none)
failed    -> (none)
cancelled -> (none)

Every arrow increments version and consumes exactly one event sequence. A retry of the same
command returns the existing result; it does not consume a second sequence. A stale event may
be observed by a consumer, but it cannot move a projection or browser view backward.
```

The state machine is monotonic in both sequence and terminality. A late running event after a
success is stale even if its payload contains a newer wall-clock timestamp; sequence and durable
ownership determine ordering.

## `docs/security/tenant-boundaries.md`

```text
Request authentication supplies tenantId. Route parameters supply a candidate jobId. Every
repository method that reads or writes a job takes both values. Event streams are keyed by the
authenticated tenant and job. A Last-Event-ID value is parsed only as a sequence cursor after
authorization. Redis channel names are derived from server-side tenant ids, never from a raw
client channel string.
```

Authorization and ordering are separate checks. A correctly ordered event from another tenant is
still an authorization failure, and a same-tenant event with a sequence gap is still an ordering
failure.

## `docs/decisions/lease-authority.md`

```text
Decision: keep the PostgreSQL job lease_id as the completion authority for this checkout.

Redis is the fast coordination lease and may expire before a worker notices. Completion opens a
transaction, locks the job row, verifies lease_id and state, writes the terminal state, and
appends the event/outbox row. A future migration may move authority to a dedicated lease table,
but it must retain the same compare-and-set property. Do not combine two independent authorities
without defining which one wins during expiry.
```

The smallest safe fix therefore keeps the existing column until a separate lease migration can
be tested with failover and clock-skew cases.

## `docs/testing/race-matrix.md`

```text
race                         required assertion
---------------------------  -----------------------------------------------
same idempotency key         one job, one key row, identical responses
key with changed body        conflict, no second job
worker loses lease            no stale terminal event or projection update
publisher retries            one visible sequence after deduplication
SSE boundary                 no duplicate and no permanent missing event
late terminal predecessor    browser and projection remain terminal
tenant collision             no cross-tenant event or job read
serialization retry          re-read durable winner before returning
```

The matrix is intentionally small. Tests should exercise the races that caused the incident,
not introduce a new messaging system or a broad rewrite of all workers.

## `docs/performance/request-path.md`

```text
The API request path is: authenticate -> canonicalize/hash request -> begin transaction -> read
or reserve idempotency key -> insert job and event/outbox atomically -> commit -> enqueue or let
the publisher deliver -> return the stored response. The browser path is: authorize -> subscribe
to live delivery -> take replay watermark -> replay durable rows -> catch up after the watermark
-> reduce monotonically.
```

The API may return before the outbox publisher has sent the queue message. It must not return
before the durable job and idempotency response have committed.

## `docs/notes/known-followups.md`

```text
Separate follow-ups include: a formal event-gap repair queue, a dedicated lease table, object
store orphan garbage collection, transactional Redis stream offsets, and a full browser offline
replay protocol. They are useful improvements but should not be folded into the minimal fix for
the incident unless a patch needs them to establish the listed invariants.
```

Keep the incident patch focused. In particular, replacing Redis Streams, changing the public API
contract, or adding a new global event service is outside the requested scope.

## `apps/api/src/middleware/auth-context.ts`

```ts
export type AuthContext = { tenantId: string; subjectId: string; scopes: string[] };

export function requireTenant(request: Request): AuthContext {
  const context = request.context?.auth;
  if (!context || !context.tenantId) throw new HttpError(401, "missing tenant context");
  return context;
}

export function requireScope(context: AuthContext, scope: string): void {
  if (!context.scopes.includes(scope)) throw new HttpError(403, "missing scope");
}
```

The route layer obtains the tenant once from authentication and passes it explicitly to every
application method. A request body, path segment, Redis channel, or Last-Event-ID value cannot
override that context. Tests should include a same-named job under two tenants and verify that
the authenticated tenant remains the only query parameter used by the repository.

## `apps/api/src/routes/job-events.ts`

```ts
export async function getJobEvents(request: Request): Promise<Response> {
  const auth = requireTenant(request);
  const jobId = request.params.jobId;
  requireScope(auth, "jobs:read");
  await jobs.assertVisible(auth.tenantId, jobId);

  const cursor = parseCursor(request.query.after);
  const after = cursor && cursor.jobId === jobId ? cursor.sequence : 0;
  const events = await eventStore.after(auth.tenantId, jobId, after, undefined);
  return Response.json(events.map(publicEvent));
}
```

This ordinary JSON endpoint is used by resync after the SSE reducer observes a gap. It is not a
shortcut around authorization. The route should reject a cursor naming another job rather than
silently treating it as a cursor for the requested job.

## `apps/api/src/infra/postgres/idempotency-repository.sql.ts`

```ts
export async function reserveKey(client: PoolClient, input: {
  tenantId: string; key: string; requestHash: string; response: unknown;
}): Promise<"inserted" | "exists"> {
  const result = await client.query(
    `INSERT INTO idempotency_keys (tenant_id, key, request_hash, response_json)
     VALUES ($1, $2, $3, $4)
     ON CONFLICT (tenant_id, key) DO NOTHING
     RETURNING key`,
    [input.tenantId, input.key, input.requestHash, JSON.stringify(input.response)],
  );
  return result.rowCount === 1 ? "inserted" : "exists";
}
```

An `ON CONFLICT DO NOTHING` reservation can avoid surfacing a unique violation, but it must be
followed by a read of the winner and a request-hash comparison. The read should happen in a
transaction that cannot observe an uncommitted partial job. It is not correct to return the
new caller's freshly generated job after the reservation lost the race.

## `apps/api/src/infra/postgres/job-write.sql`

```sql
-- All statements run on one client between BEGIN and COMMIT.
SELECT id, state, version, next_event_sequence, lease_id
  FROM jobs
 WHERE tenant_id = $1 AND id = $2
 FOR UPDATE;

UPDATE jobs
   SET state = $3,
       version = version + 1,
       next_event_sequence = next_event_sequence + 1,
       lease_id = $4,
       updated_at = clock_timestamp()
 WHERE tenant_id = $1
   AND id = $2
   AND state = $5
   AND (lease_id = $6 OR $6 IS NULL)
 RETURNING id, tenant_id, state, version, next_event_sequence, lease_id, updated_at;
```

The application may choose a row lock or a conditional update, but it must inspect the returned
row count and never build an event after a zero-row update. If both are used, the lock makes the
decision easy to reason about and the condition provides an additional ownership assertion.

## `packages/events/src/dedup-store.ts`

```ts
export class EventDedupStore {
  constructor(private readonly client: RedisClientType) {}

  async firstSeen(event: JobEvent): Promise<boolean> {
    const key = `event-seen:${event.tenantId}:${event.jobId}:${event.sequence}`;
    const result = await this.client.set(key, event.eventId, { NX: true, EX: 86400 });
    return result === "OK";
  }
}
```

Redis deduplication improves consumer efficiency but is not the durable source of truth: its TTL
and eviction policy can remove an old marker. The projector's monotonic sequence check remains
mandatory, and a replay after a restart must remain safe even if the Redis marker is gone.

## `apps/worker/src/queue-consumer.ts`

```ts
export async function consume(message: QueueMessage, deps: WorkerDeps): Promise<void> {
  const parsed = jobMessageSchema.parse(message);
  const lease = await deps.leases.acquire(
    `job:${parsed.tenantId}:${parsed.jobId}`,
    deps.workerId,
    deps.config.leaseTtlMs,
  );
  if (!lease) return;
  const controller = new AbortController();
  const renewer = deps.makeRenewer(parsed, controller);
  renewer.start();
  try {
    const running = await deps.commands.markRunning(parsed, deps.workerId);
    if (!running.ok) return;
    const output = await deps.renderer.render({ ...parsed, signal: controller.signal });
    if (renewer.hasLost()) {
      await deps.objects.scheduleOrphanCleanup(output.objectKey);
      return;
    }
    await deps.commands.complete(parsed, deps.workerId, output);
  } finally {
    renewer.stop();
    await deps.leases.release(`job:${parsed.tenantId}:${parsed.jobId}`, deps.workerId);
  }
}
```

The early renewer check is a latency optimization only. `commands.complete` must still perform
the authoritative database check because lease loss can occur between the check and the commit.
The queue acknowledgement policy should retain a failed or lost-lease message for a recovery
worker rather than acknowledging a job as successfully completed.

## `apps/worker/src/recovery-queue.ts`

```ts
export async function requeueLostLease(job: Job, queue: JobQueue): Promise<void> {
  if (job.state === "succeeded" || job.state === "failed" || job.state === "cancelled") return;
  await queue.enqueue({
    tenantId: job.tenantId,
    jobId: job.id,
    reason: "lease-lost",
    attempt: job.attempt + 1,
  });
}
```

Recovery is a separate transition policy. It cannot reopen a terminal job, and it must not emit a
new `running` event until a new owner has been acquired and the current job row has been read.

## `packages/events/src/event-store.ts` additional methods

```ts
export class EventStore {
  async maxSequence(tenantId: string, jobId: string): Promise<number> {
    const result = await this.pool.query<{ max: number | null }>(
      `SELECT max(sequence)::bigint AS max FROM job_events WHERE tenant_id = $1 AND job_id = $2`,
      [tenantId, jobId],
    );
    return Number(result.rows[0]?.max ?? 0);
  }

  async after(tenantId: string, jobId: string, after: number, through?: number): Promise<JobEvent[]> {
    const result = await this.pool.query<JobEvent>(
      `SELECT event_id AS "eventId", tenant_id AS "tenantId", job_id AS "jobId",
              sequence, occurred_at AS "occurredAt", type, state, payload
         FROM job_events
        WHERE tenant_id = $1 AND job_id = $2 AND sequence > $3
          AND ($4::bigint IS NULL OR sequence <= $4)
        ORDER BY sequence`,
      [tenantId, jobId, after, through ?? null],
    );
    return result.rows;
  }
}
```

The upper bound is passed as a SQL value rather than interpolated. A nullable upper bound means
the catch-up query includes all rows after the latest cursor. The event store does not infer a
tenant from `jobId`, and it does not return rows in arrival order.

## `packages/events/src/consumer.ts`

```ts
export async function consumeEvent(raw: unknown, deps: ConsumerDeps): Promise<void> {
  const event = jobEventSchema.parse(raw);
  if (!(await deps.authorizer.canProject(event.tenantId, event.jobId))) return;
  await inTransaction(deps.pool, async (client) => {
    const status = await applyEvent(client, event);
    if (status === "gap") {
      await deps.gaps.record(client, event.tenantId, event.jobId, event.sequence);
    }
  });
}
```

Projection processing can be retried. A duplicate is a no-op, a contiguous event advances the
row, and a gap is recorded for repair. The consumer does not skip a gap merely because a later
event has a newer timestamp.

## `apps/web/src/dashboard/job-view.ts`

```ts
export type JobView = {
  tenantId: string;
  id: string;
  state: JobState;
  sequence: number;
  result?: Record<string, unknown>;
  needsResync: boolean;
};

export function mergeInitialJob(initial: Job, prior: JobView | undefined): JobView {
  if (!prior || prior.tenantId !== initial.tenantId || prior.id !== initial.id) {
    return { tenantId: initial.tenantId, id: initial.id, state: initial.state, sequence: initial.version, needsResync: false };
  }
  if (prior.sequence >= initial.version || terminal.has(prior.state)) return prior;
  return { ...prior, state: initial.state, sequence: initial.version, needsResync: false };
}
```

Initial HTTP data and live SSE data may arrive in either order. The UI state uses the same
sequence/terminal rule for both paths so a late HTTP response cannot reopen a terminal row.

## `apps/web/src/dashboard/job-reducer.test.ts`

```ts
it("ignores a delayed running event after success", () => {
  const success = reduceJob({ tenantId: "t", id: "j", state: "succeeded", sequence: 3, needsResync: false }, event("job.succeeded", 3));
  const delayed = reduceJob(success, event("job.running", 2));
  expect(delayed).toEqual(success);
});

it("flags a sequence gap instead of inventing state", () => {
  const state = reduceJob({ tenantId: "t", id: "j", state: "running", sequence: 1, needsResync: false }, event("job.succeeded", 3));
  expect(state.needsResync).toBe(true);
  expect(state.state).toBe("running");
});
```

The reducer test is intentionally independent of network timing. The server-side replay test
proves delivery; this test proves the view cannot regress if a duplicate or delayed callback is
still delivered by a browser event source.

## `db/migrations/007_event-gap-records.sql`

```sql
CREATE TABLE event_gaps (
  tenant_id text NOT NULL,
  job_id text NOT NULL,
  expected_sequence bigint NOT NULL,
  observed_sequence bigint NOT NULL,
  first_seen_at timestamptz NOT NULL DEFAULT now(),
  last_seen_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, job_id, expected_sequence),
  FOREIGN KEY (tenant_id, job_id) REFERENCES jobs (tenant_id, id)
);
```

This table is a possible follow-up for durable gap repair. It is not needed to make the minimal
write-side transaction atomic, but it illustrates why a consumer should report a gap rather than
pretend a missing sequence is complete.

## `db/migrations/008-outbox-claim.sql`

```sql
ALTER TABLE outbox
  ADD COLUMN claimed_by text,
  ADD COLUMN claimed_at timestamptz,
  ADD COLUMN published_at timestamptz;

-- A future publisher can use this pattern after selecting rows in a short transaction.
-- SELECT ... FOR UPDATE SKIP LOCKED; update claimed_by; commit; publish; mark published.
```

Claiming outbox rows can improve parallelism, but it does not change at-least-once semantics.
The publisher must continue to tolerate a crash between external delivery and the durable
published marker.

## `docs/architecture/sequence-allocation.md`

```text
For every transition, lock jobs(tenant_id, id), read next_event_sequence, and use that exact
number for the event. Update the job projection and increment next_event_sequence in the same
transaction. Do not allocate from a process-local counter, Redis INCR, wall-clock timestamp, or
the worker's stale copy of version. The unique index is a guardrail, not a sequencer.
```

The first queued event may use sequence 1 while the job version is 1. The exact initial value is
less important than the invariant that projection version and event sequence advance together
and that every retry returns the already committed result.

## `docs/operations/worker-failure.md`

```text
If a worker dies after claiming a job, the lease expires and the recovery poller sees the job as
running with an expired lease. The poller may enqueue a retry. If the first worker later returns,
its completion is rejected by the lease authority. Operators should expect an uploaded orphan
object and should not manually write a success event to make the dashboard look complete.
```

A recovery poller must lock the row before deciding whether it is eligible, and it must not
change a terminal state back to running. Orphan cleanup is keyed by object checksum and is safe
to repeat.

## `docs/api/idempotency.md`

```text
Idempotency-Key is required for POST /v1/jobs. Maximum length is 128 bytes after UTF-8
validation. The key is scoped to the authenticated tenant. The canonical request hash excludes
the key itself. A matching retry returns the original HTTP status and response body. A mismatch
returns 409 with a stable error code. The record is retained for the documented replay window.
```

Input validation belongs before the database transaction, but the final request hash and
response must be decided inside the transaction that inserts the durable key. The key cannot be
reserved in Redis alone because Redis loss would permit a duplicate job.

## `docs/api/event-stream.md`

```text
The SSE event id is jobId:sequence. The stream sends retry advice but does not promise that a
client receives exactly one network packet. Clients must tolerate duplicate ids. The server
must send events in sequence order for one authorized job, and a reconnect must replay from the
last accepted sequence. If a gap is detected, the client requests a full JSON resync.
```

SSE framing, TCP delivery, and Redis delivery are all at-least-once boundaries. The monotonic
cursor is the small piece of state that makes them compose safely.

## `docs/testing/transaction-fixtures.md`

```ts
export async function runConcurrent<T>(first: () => Promise<T>, second: () => Promise<T>): Promise<[T, T]> {
  const barrier = makeBarrier(2);
  return Promise.all([
    (async () => { await barrier.arrive(); return first(); })(),
    (async () => { await barrier.arrive(); return second(); })(),
  ]) as Promise<[T, T]>;
}

export async function assertSingleJob(pool: Pool, tenantId: string, key: string): Promise<void> {
  const rows = await pool.query(`SELECT count(*)::int AS count FROM jobs j JOIN idempotency_keys i ON i.response_json->>'id' = j.id WHERE j.tenant_id = $1 AND i.key = $2`, [tenantId, key]);
  expect(rows.rows[0].count).toBe(1);
}
```

Race fixtures should use barriers and real database constraints. Sleeping for a guessed number
of milliseconds is not a deterministic concurrency test and can pass while still leaving the
unique-key race unprotected.

## `docs/decisions/minimal-incident-fix.md`

```text
In scope: tenant-scoped idempotency reservation and winner re-read; lock-and-append state
transitions; lease check in completion; subscribe-before-replay plus a catch-up read; monotonic
browser reducer; focused race tests.

Out of scope: replacing Redis, migrating all jobs to a new lease table, changing event schemas,
rewriting the dashboard transport, or building a general workflow engine.
```

The incident fix should be reviewable as a small set of application, repository, route, and test
changes. Follow-ups can be tracked separately after the invariants are protected.

## `apps/api/src/infra/postgres/advisory-locks.ts`

```ts
export async function lockTenantKey(client: PoolClient, tenantId: string, key: string): Promise<void> {
  // This lock is an optional optimization for hot keys. The unique constraint
  // remains the correctness mechanism and must still be present.
  await client.query("SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", [`${tenantId}:${key}`]);
}

export async function lockJob(client: PoolClient, tenantId: string, jobId: string): Promise<void> {
  await client.query("SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", [`${tenantId}:${jobId}`]);
}
```

An advisory lock can reduce duplicate work before a row exists, but it must not replace the
unique index or row lock. Hash collisions are possible, and a process crash releases the lock
without rolling back a committed idempotency response. Use it only as a performance aid around
the durable transaction.

## `apps/api/src/application/event-command.ts`

```ts
export async function transitionJob(input: TransitionInput, deps: Deps): Promise<Result<Job, DomainError>> {
  return inTransaction(deps.pool, async (client) => {
    const current = await deps.jobs.lock(client, input.tenantId, input.jobId);
    if (!current) return { ok: false, error: { kind: "not-found" } };
    if (terminal.has(current.state)) return { ok: false, error: { kind: "conflict", message: "job is terminal" } };
    if (!assertTransition(current.state, input.nextState)) {
      return { ok: false, error: { kind: "conflict", message: "invalid transition" } };
    }
    const eventSequence = current.nextEventSequence;
    const updated = await deps.jobs.applyTransition(client, current, input.nextState);
    const event = deps.events.build(input, updated, eventSequence);
    await deps.events.append(client, event);
    await deps.outbox.append(client, event);
    return { ok: true, value: updated };
  });
}
```

The event sequence is captured from the locked current row and is included in the event before
the transaction commits. It is not recomputed from `version` after another query. If a domain
command returns a conflict, it emits no event and does not increment the projection.

## `apps/api/src/infra/postgres/outbox-repository.ts` additional queries

```ts
export async function unpublished(pool: Pool, limit: number): Promise<OutboxRow[]> {
  const result = await pool.query<OutboxRow>(
    `SELECT id, event_id AS "eventId", tenant_id AS "tenantId", job_id AS "jobId",
            sequence, payload
       FROM outbox
      WHERE published_at IS NULL
      ORDER BY id
      LIMIT $1`,
    [limit],
  );
  return result.rows;
}

export async function markPublished(pool: Pool, id: number): Promise<void> {
  await pool.query(`UPDATE outbox SET published_at = clock_timestamp() WHERE id = $1 AND published_at IS NULL`, [id]);
}
```

The conditional marker makes duplicate publisher workers harmless, but the external stream
write is still allowed to happen more than once. Consumers deduplicate by durable sequence and
do not treat a duplicate message as a state transition.

## `packages/events/src/terminal-state.ts`

```ts
export const terminalStates = ["succeeded", "failed", "cancelled"] as const;
export type TerminalState = (typeof terminalStates)[number];

export function isTerminal(state: string): state is TerminalState {
  return (terminalStates as readonly string[]).includes(state);
}

export function canApplyToProjection(current: { state: string; version: number }, event: JobEvent): boolean {
  if (event.sequence <= current.version) return false;
  if (isTerminal(current.state)) return false;
  return event.sequence === current.version + 1;
}
```

Terminality is a domain property, not a string comparison scattered across route handlers. Keep
the same helper in the projector and browser package, or encode the rule in a shared contract so
that a delayed event cannot be accepted in one consumer and rejected in another.

## `docs/operations/incident-checklist.md`

```text
When the dashboard shows a terminal job as active:

* capture the tenant, job id, current projection version, and durable event sequences;
* check for duplicate idempotency rows and duplicate outbox deliveries;
* compare each event's tenant/job/sequence with the projection;
* inspect lease ownership at the time of the terminal event;
* do not delete an event, manually rewrite the browser state, or raise a retry limit first;
* after the durable fix, replay the authorized stream and verify the browser reducer.
```

The durable sequence list is the primary forensic record. Wall-clock timestamps and delivery
order can explain the symptom but cannot establish which state transition was authoritative.

## `apps/api/src/application/idempotency-retry.ts`

```ts
export async function createWithWinnerRecovery(input: CreateInput, deps: Deps): Promise<Job> {
  for (let attempt = 1; attempt <= 2; attempt += 1) {
    try {
      return await deps.create(input);
    } catch (error) {
      if (!isUniqueViolation(error) || attempt === 2) throw error;
      const winner = await deps.idempotency.read(input.tenantId, input.idempotencyKey);
      if (!winner) continue;
      if (winner.requestHash !== hash(input)) throw new ConflictError("idempotency key reused with different request");
      return winner.response;
    }
  }
  throw new Error("unreachable");
}
```

The recovery path is shown to make the race explicit. A production implementation may instead
reserve with `ON CONFLICT`, but every path after a unique conflict must compare the durable
winner before returning. It must not generate a new job id outside the transaction and then
assume the failed insert rolled back another request.

## `apps/api/src/infra/postgres/row-mappers.ts`

```ts
export function mapJob(row: Record<string, unknown>): Job {
  return {
    id: String(row.id),
    tenantId: String(row.tenant_id),
    state: row.state as JobState,
    version: Number(row.version),
    nextEventSequence: Number(row.next_event_sequence),
    leaseId: row.lease_id === null ? null : String(row.lease_id),
    createdAt: new Date(String(row.created_at)),
    updatedAt: new Date(String(row.updated_at)),
  };
}

export function mapEvent(row: Record<string, unknown>): JobEvent {
  return jobEventSchema.parse({
    eventId: row.event_id,
    tenantId: row.tenant_id,
    jobId: row.job_id,
    sequence: Number(row.sequence),
    occurredAt: new Date(String(row.occurred_at)).toISOString(),
    type: row.type,
    state: row.state,
    payload: row.payload,
  });
}
```

Mapping code is deliberately explicit about numeric sequence values. PostgreSQL drivers can
return `bigint` columns as strings, and comparing a string sequence lexicographically can make
sequence 10 appear older than sequence 2. The same normalized type is used by the projector,
SSE route, and browser contract.

## `docs/operations/replay-validation.sql`

```sql
SELECT j.tenant_id,
       j.id AS job_id,
       j.state AS projection_state,
       j.version AS projection_version,
       coalesce(max(e.sequence), 0) AS event_version,
       count(e.sequence) AS event_count,
       count(e.sequence) = max(e.sequence) AS has_no_simple_gap
  FROM jobs j
  LEFT JOIN job_events e
    ON e.tenant_id = j.tenant_id AND e.job_id = j.id
 WHERE j.tenant_id = $1 AND j.id = $2
 GROUP BY j.tenant_id, j.id, j.state, j.version;
```

This diagnostic query is tenant-scoped and uses a qualified join. It is not a replacement for a
transactional repair command: a concurrent transition can change the result between checks.
Use it to collect evidence before replaying an authorized stream.

## `docs/architecture/commit-boundary.txt`

```text
The commit boundary for a state transition includes:

  locked current job row
      | validate state and ownership
      v
  updated job projection + allocated sequence
      | append event and outbox payload
      v
  one PostgreSQL COMMIT

Redis delivery and object-store visibility occur after this boundary. A process may crash on
either side; recovery must be safe in both cases. No event is published from a stale in-memory
job object after the transaction has released its lock.
```

Keeping this boundary visible in the code review is more valuable than adding a broad retry
framework. The worker and API may share repository helpers, but they must not split the job
update and event append across independent connections.
