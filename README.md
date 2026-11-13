# Black Mesa As Built

bmab maps the Black Mesa Research Facility as Half-Life, Black Mesa and their mods build it. You look through each map in 3D,
mark what you find, and send it to the redlines, the shared record everyone works on.

## Getting started

1. Download bmab for Windows or Linux from this repository's [Releases](../../releases/latest), unpack it and start `bmab`.
   Windows may warn about an unknown publisher: choose More info, then Run anyway. On Windows bmab opens in a window of its own,
   on Linux in your browser. Quit on its front page ends it.
2. On the front page, sign in with GitHub, so that what you send is yours.
3. Under Your games, copy in the games you have. bmab finds them in your Steam libraries; for the rest, the front page says
   where to get each one.
4. Open an edition under Maps, by edition, and pick a map. Its view takes a moment to build the first time.

To update bmab, unpack the new download where you unpacked the old one, replacing its files: your games, the redlines and
your edits stay.

To remove bmab, delete its folder and the bmab folders in your settings: `%APPDATA%\bmab` and `%LOCALAPPDATA%\bmab` on
Windows, `~/.config/bmab` on Linux (your GitHub sign-in is kept there).

From the source instead: with Python 3.11 or later, install numpy, pillow and scipy (and pywebview on Windows), then run
`python -m bmab app` in this folder.

## Sending your work

Your edits are saved on your computer as you make them. Commit and Push send them to the redlines, or, if you can't push
there, as a pull request that bmab opens for you.

What others have sent comes in when bmab starts and before each Push. Where someone else changed something you changed too,
the map's page (or the register's) shows both, and you keep yours or take theirs before committing.

Nothing from the games leaves your computer: bmab reads its own copy of them, and the redlines hold only what people say.

## Licence

[AGPL-3.0-or-later](LICENSE)
