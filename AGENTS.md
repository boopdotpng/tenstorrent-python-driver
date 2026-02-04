# agents
Write excellent, concise code that solves problems elegantly.

## philosophy
**Every line must earn its keep.** Prefer readability over cleverness. If carefully designed, 10 lines can have the impact of 1000.

Key principles:
- **Balance file size**: Avoid 1000+ line files unless everything truly belongs together. Break into logical modules, but don't create file-spam (single-function files)
- **Validate changes**: Run `cargo check` (or project equivalent) after modifying library code
- **Trust your code**: Skip excessive error-checking (e.g., constant try-catches in Python)
- **Compress wisely**: One-line when readable. Example: `if not path.is_file(): continue`

## style
- **2-space indentation** everywhere
- Match existing code style

## tenstorrent work
When working in tenstorrent repos or talking about tenstorrent stuff, I might ask you do "add that to the docs". The docs are at /home/boop/tenstorrent/boop-docs/. It's split up by folder, so if what you're adding to the docs doesn't match the existing categories, create a new folder and add a markdown file inside. Don't worry about reading through the docs and inserting stuff, just make a brand new markdown file each time. The 'human' folder is read-only for you. 

All tenstorrent repos are accessible to you at ~/tenstorrent/. Examples include tt-metal, tt-kmd, tt-umd, etc. 

## environment
Host: Fedora 43 Server 
Tools: `sg` (ast-grep), `uv`, `bun`
