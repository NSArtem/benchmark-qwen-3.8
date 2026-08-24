#!/bin/zsh

# Read-only package roots from the existing uv archive.  The checked-in
# project venv is intentionally bare in this workspace; these paths expose
# the already-pinned MLX 0.32.1 / mlx-lm revision without syncing or changing
# the canonical project.
export DYLD_LIBRARY_PATH="/Users/artem/.cache/uv/archive-v0/eGyaoXUFk5Atz50c/mlx/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
export PYTHONPATH="/Users/artem/.cache/uv/archive-v0/G2l4z9UFcvvWCQu9:/Users/artem/.cache/uv/archive-v0/GD4ii1Q6IhKy2-zq:/Users/artem/.cache/uv/archive-v0/qMAZP8bPFgSmKXxA:/Users/artem/.cache/uv/archive-v0/RT70_HqWD0CxoVCR:/Users/artem/.cache/uv/archive-v0/hO2qLeivOkEOaNrV:/Users/artem/.cache/uv/archive-v0/K5LpWUV7xzmRfZaD:/Users/artem/.cache/uv/archive-v0/ZIGDlBvajXfnQW3n:/Users/artem/.cache/uv/archive-v0/uDYMlmjvFNiFQwVc:/Users/artem/.cache/uv/archive-v0/Yu5ThqJ44XDhsbFb:/Users/artem/.cache/uv/archive-v0/SGWkFFHyYREC09Ct:/Users/artem/.cache/uv/archive-v0/Iaw-QIOQQcZW_Bm5:/Users/artem/.cache/uv/archive-v0/3ZWeHnL0IT-RYa6p:/Users/artem/.cache/uv/archive-v0/1mg8RolnYpaf9N2J:/Users/artem/.cache/uv/archive-v0/q4jQ43v4hVrVs8nr:/Users/artem/.cache/uv/archive-v0/XmQw0QZk8ENWwtCr:/Users/artem/.cache/uv/archive-v0/sjriM6ZSY2NC_FWZ:/Users/artem/.cache/uv/archive-v0/Auwr3_IDYS89a7cD:/Users/artem/.cache/uv/archive-v0/_kxk4kZmO4EsfxSl:/Users/artem/.cache/uv/archive-v0/a1cmsQo-uPxIR5By:/Users/artem/.cache/uv/archive-v0/O_7oE8BhObdBzx-t:/Users/artem/.cache/uv/archive-v0/YuxeeM7GOeOJHCeX:/Users/artem/.cache/uv/archive-v0/N0VnZSQnHf85S0aj:/Users/artem/.cache/uv/archive-v0/jWkMQTwxR2tM6Xq3:/Users/artem/.cache/uv/archive-v0/ivKF3Mcnm-zX6uPE:/Users/artem/.cache/uv/archive-v0/MyIT8p5UMcWV4tzu:/Users/artem/.cache/uv/archive-v0/iqVMSNOsP8-iFz45:/Users/artem/.cache/uv/archive-v0/zRmLo4OgucElRHYB:/Users/artem/.cache/uv/archive-v0/fKrR5pNqlipeXu1C:/Users/artem/.cache/uv/archive-v0/xpO_XfU5fud_p91n:/Users/artem/.cache/uv/archive-v0/VkQPCYXhAQHca2vn:/Users/artem/.cache/uv/archive-v0/TZphLg32C4qM4a0y:/Users/artem/.cache/uv/archive-v0/KKETfOlQfJRArH3h${PYTHONPATH:+:$PYTHONPATH}"
typeset -a stage4_archive_roots
stage4_archive_roots=(/Users/artem/.cache/uv/archive-v0/*)
export PYTHONPATH="${PYTHONPATH}:${(j.:.)stage4_archive_roots}"
export UV_CACHE_DIR="/private/tmp/qwen38-stage4-uv-cache"
