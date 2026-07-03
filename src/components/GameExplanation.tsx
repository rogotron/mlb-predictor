import type { GamePrediction } from '../data/mockModelData';

export function buildGameExplanation(game: GamePrediction) {
  return game.explanation;
}

export function GameExplanation({ game }: { game: GamePrediction }) {
  const winner = game.homeWinProbability >= game.awayWinProbability ? game.homeTeam : game.awayTeam;
  const winnerProb = Math.max(game.homeWinProbability, game.awayWinProbability);
  const mainFactor = game.topFactors[0];
  const separator = game.topFactors[1] ?? game.topFactors[0];
  const confidenceLimit = game.riskFactors[0] ?? 'Late lineup, bullpen, and run-environment variance keep the edge from becoming automatic.';
  const sections = [
    {
      label: 'Why the model favors the pick',
      copy: `${winner.name} grades as the model side at ${winnerProb}% because the current slate points to the cleaner win path.`,
    },
    {
      label: 'Key separator',
      copy: mainFactor ? `${mainFactor.name}: ${mainFactor.description}` : buildGameExplanation(game),
    },
    {
      label: 'What limits confidence',
      copy: separator && separator !== mainFactor
        ? `${confidenceLimit} The next-most important separator is ${separator.name.toLowerCase()}.`
        : confidenceLimit,
    },
  ];

  return (
    <section className="glass-card p-5">
      <p className="section-kicker">Written model read</p>
      <h3 className="mt-1 text-lg font-semibold text-slate-950">Game explanation</h3>
      <div className="mt-4 grid gap-3">
        {sections.map((section) => (
          <div className="rounded-lg border border-slate-200/75 bg-white/55 p-3" key={section.label}>
            <p className="text-xs font-black uppercase tracking-[0.14em] text-slate-500">{section.label}</p>
            <p className="mt-2 text-sm leading-6 text-slate-700">{section.copy}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
