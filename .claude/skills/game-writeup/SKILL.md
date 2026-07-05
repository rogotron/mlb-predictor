---
name: game-writeup
description: >-
  Write punchy, analytical per-game summaries from a Diamond Forecast slate
  prediction (public/slates/{date}.json). Use this whenever the user asks for a
  game writeup, matchup recap, capsule, blurb, preview, or "what does the model
  say about tonight's game" for one or more games on the slate, even if they
  don't say the word "writeup". The hard rule this skill enforces: narrate only
  from the slate's real model-derived factors, because the deployed pregame_safe
  model has no batter-vs-pitcher, weather, umpire, or home-field-edge features.
---

# Game writeup

Turn a single game object from the slate JSON into a short, confident scouting
blurb. The model already did the analysis; your job is to translate its numbers
into prose a sharp baseball reader would respect.

## Read the real data first

Predictions live in `public/slates/{date}.json` (written by
`scripts/build_static_slate.py`). If the user names a date, read that file; if
not, use today's slate. Find the game by team abbreviation in `game.away.abbr`
/ `game.home.abbr` or by team name.

Every game object has these fields you narrate from (real names, use them):

- `game.prediction` — `winner`, `homeProb`, `awayProb`, `confLabel`
  (`MARGINAL`/`LOW`/`MODERATE`/`HIGH`), `total`, `awayRuns`/`homeRuns`,
  `awayScore`/`homeScore`.
- `game.factors[]` — the model's real drivers, already ranked by importance.
  Each is `{name, source, pct, note}`. **This array is your evidence base.**
- `game.pitchers.away` / `game.pitchers.home` — `name`, `hand`, `era`, `whip`,
  `k9`, `rec`, `last` (last start line).
- `game.stats[]` — head-to-head rows (`R/GAME (L10)`, `RECORD`, `WIN PCT`,
  `SP ERA`, `SP WHIP`, `SP K/9`, `DIV STANDING`), each `{stat, away, home}`.
- `game.game` — `venue`, `firstPitch`.

`game.prediction.drivers[]` holds pre-written sentences. Treat them as raw
material, not finished copy. Build your own tighter lines from the factors and
the numbers.

## The one hard constraint

`game.factors[]` is the complete, honest list of what moved the pick. Attribute
the verdict only to what is in that array. For the deployed `pregame_safe`
model those groups are:

Starting Pitcher Quality, Bullpen Quality + Load, Run Production, Recent Form
(L10/L20), Rest & Availability, Home/Away Record, Park Factors.

Do not write about inputs the model does not have. Specifically never mention:

- batter-vs-pitcher / BvP history
- weather, wind, temperature, rain
- umpire tendencies or strike zones
- a home-field "edge", "boost", or "advantage" as a reason for the pick
- lineup platoon splits or specific batter-vs-hand matchups

These were audited out of the default model as leakage or as never-implemented.
Inventing them makes the writeup lie about how the number was produced. The only
legitimate "home" references are the literal `game.venue` and the Home/Away
Record split (which is a real factor). If a sentence you drafted names anything
on the forbidden list, delete it.

## Voice

Write like a confident analyst who respects the reader's time.

- **Verdict first.** Open with the pick and its win probability. The reader
  should know the call before the evidence.
- **Numbers are the argument.** Cite the factor percentages and the concrete
  stats (ERA, K/9, R/G, records). A claim without a number is filler.
- **Punchy, not padded.** Short declaratives. Cut hedging clichés ("should be a
  good one", "anything can happen", "on paper").
- **No em dashes.** Use periods, commas, or colons. This is a house style rule.
- **Honest about conviction.** If `confLabel` is `MARGINAL` or `LOW`, say so
  plainly. Do not dress a coin flip as a lock.

## Structure

Default to four to six sentences per game (shorter if the user wants a one-liner,
longer only if asked):

1. Verdict: winner, win probability, and a read on confidence.
2. Lead evidence: the top one or two `factors[]` by `pct`, tied to the real
   numbers behind them (starter lines, R/G, records).
3. Supporting evidence: the next factor or the season context that separates the
   teams.
4. Close: projected score and total, plus an honest note if the edge is thin.

## Example

Input: the Mets at Braves game. `prediction.winner` = Atlanta Braves,
`homeProb` 51, `confLabel` MARGINAL, `total` 8.8, projected 5-4. Top factors:
Starting Pitcher Quality 31.8 (`Nolan McLean vs Martín Pérez`), Bullpen Quality
+ Load 22.4, Run Production 19.7 (`3.2 vs 4.5 R/G last 10`). Records 36-53 and
52-35.

Output:

> Braves by a hair, 51 percent, and the model wants you to know it is barely a
> lean. Starting pitcher quality is the heaviest input at 31.8 percent of the
> model, and it nearly cancels out: Nolan McLean (3.78 ERA, 10.7 K/9) for the
> Mets against Martín Pérez (3.27 ERA) for Atlanta. The separation is the season
> underneath the arms. Atlanta is 52-35 and scoring 4.5 a game over its last 10;
> the Mets are 36-53 and stuck at 3.2, and bullpen and run production, the next
> two factors at 22.4 and 19.7 percent, tilt the same way. Projected 5-4 Atlanta,
> total 8.8. This is a low-conviction play, so treat it like the near-even game
> it is.

Note what that example does not do: no weather, no bullpen-vs-a-specific-hitter,
no "Braves get the home crowd." Just the factors the model actually used, with
the numbers that back them.

## Multiple games

If the user asks for the whole slate or several games, write one blurb per game
under a short header (matchup and first pitch), keep the voice identical, and do
not invent a cross-game "game of the day" narrative unless asked. Sort by model
confidence (widest `homeProb`/`awayProb` gap first) so the strongest reads lead.
