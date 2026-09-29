# Demo recordings

`make demo` records `demo/out/mlx-demo.mp4` from the [VHS](https://github.com/charmbracelet/vhs)
tapes here (one per segment), then joins them with ffmpeg.

Needs `vhs` (with `ttyd` and `ffmpeg`), the CLI built (`make cli`), the models
the tapes use (downloaded on first use), Emacs, and a
[SLY](https://github.com/joaotavora/sly) checkout named by `$SLY_DIR` for the
editor segment.
