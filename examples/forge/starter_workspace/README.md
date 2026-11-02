# wordstats (starter workspace for Forge)

A tiny module with a passing test suite. Point Forge at a copy of this folder and give it the task in
`../extend_wordstats.md`:

```bash
cp -r examples/forge/starter_workspace /tmp/wordstats
nimlab --mock forge run --task-file examples/forge/extend_wordstats.md --workspace /tmp/wordstats
```
