import { useEffect, useMemo, useState, type ReactElement } from 'react';
import { Activity, BarChart3, CalendarDays, RefreshCw, SlidersHorizontal, TriangleAlert } from 'lucide-react';
import { AccuracyOverview } from './components/AccuracyOverview';
import { CalibrationChart } from './components/CalibrationChart';
import { ConfidenceBadge } from './components/ConfidenceBadge';
import { FactorImpactList } from './components/FactorImpactList';
import { GameDetailPanel } from './components/GameDetailPanel';
import { MatchupCard } from './components/MatchupCard';
import { ModelHealthCard } from './components/ModelHealthCard';
import { PerformanceChart } from './components/PerformanceChart';
import { PowerRankingsCard } from './components/PowerRankingsCard';
import { RecentPredictionsTable } from './components/RecentPredictionsTable';
import { Sidebar, type NavKey } from './components/Sidebar';
import { SummaryMetricCard } from './components/SummaryMetricCard';
import { TeamIdentity } from './components/TeamIdentity';
import { TopNav } from './components/TopNav';
import { WinProbabilityBar } from './components/WinProbabilityBar';
import {
  accuracyRecords,
  buildPowerRankingsFromPredictions,
  calibrationData,
  fallbackScheduleGames,
  generateAccuracyRecordForGame,
  generateMockPredictionForGame,
  modelFactors,
  performanceTrend,
  predictionWinner,
} from './data/mockModelData';
import type { GamePrediction, ModelFactor } from './data/mockModelData';
import { fetchMlbSchedule, fetchMlbScheduleRange } from './services/mlbScheduleApi';
import type { MlbGame } from './services/mlbScheduleApi';
import {
  buildPredictionFromModelSlate,
  fetchModelSlate,
  modelSummaryToFactors,
  type ModelSlateStatus,
  type SlateModelSummary,
  type SlateProvenance,
} from './services/modelSlate';
import { ProvenanceBar } from './components/ProvenanceBar';

const confidenceOrder = ['All', 'High', 'Medium', 'Low'] as const;
const edgeFilters = [
  'All games',
  'Highest edge',
  'Highest confidence',
  'Home favorites',
  'Underdog value',
  'Pitcher advantage',
] as const;

const compactNavItems: Array<{ key: NavKey; label: string }> = [
  { key: 'dashboard', label: 'Dashboard' },
  { key: 'games', label: "Today's Games" },
  { key: 'game-detail', label: 'Game Detail' },
  { key: 'accuracy', label: 'Accuracy' },
  { key: 'factors', label: 'Factors' },
  { key: 'settings', label: 'Settings' },
];

type ConfidenceFilter = (typeof confidenceOrder)[number];
type EdgeFilter = (typeof edgeFilters)[number];
type ViewMode = 'probability' | 'score';
type AccuracyWindow = '7 days' | '30 days' | 'Season to date';
type ScheduleSource = 'live' | 'fallback';

function easternDateString(value = new Date()) {
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/New_York',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).formatToParts(value);
  const byType = Object.fromEntries(parts.map((part) => [part.type, part.value]));
  return `${byType.year}-${byType.month}-${byType.day}`;
}

function browserToday() {
  return easternDateString();
}

function formatDateLabel(date: string) {
  const [year, month, day] = date.split('-').map(Number);
  return new Date(year, month - 1, day).toLocaleDateString(undefined, {
    weekday: 'short',
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });
}

function lastUpdatedLabel() {
  return new Intl.DateTimeFormat('en-US', {
    hour: 'numeric',
    minute: '2-digit',
    timeZone: 'America/New_York',
    timeZoneName: 'short',
  }).format(new Date());
}

// Formats the slate's data-as-of ISO timestamp (build time), so the header
// reflects when the data was actually assembled — not the browser's clock.
function formatAsOfLabel(iso: string) {
  const parsed = new Date(iso);
  if (Number.isNaN(parsed.getTime())) return iso;
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
    timeZone: 'America/New_York',
    timeZoneName: 'short',
  }).format(parsed);
}

function shiftDate(date: string, days: number) {
  const [year, month, day] = date.split('-').map(Number);
  return new Date(Date.UTC(year, month - 1, day + days)).toISOString().slice(0, 10);
}

function shiftDateString(date: string, days: number) {
  return shiftDate(date, days);
}

function applyGameFilters(
  games: GamePrediction[],
  confidence: ConfidenceFilter,
  edgeFilter: EdgeFilter,
) {
  let next = [...games];
  if (confidence !== 'All') {
    next = next.filter((game) => game.confidenceLabel === confidence);
  }

  if (edgeFilter === 'Highest edge') {
    next.sort((a, b) => Math.abs(b.modelEdge) - Math.abs(a.modelEdge));
  } else if (edgeFilter === 'Highest confidence') {
    next.sort((a, b) => b.confidenceScore - a.confidenceScore);
  } else if (edgeFilter === 'Home favorites') {
    next = next.filter((game) => game.homeWinProbability > game.awayWinProbability);
  } else if (edgeFilter === 'Underdog value') {
    next = next.filter((game) => game.modelEdge >= 3);
  } else if (edgeFilter === 'Pitcher advantage') {
    next = next.filter((game) => game.topFactors.some((factor) => factor.category === 'Pitching'));
  }

  return next;
}

function SourceBanner({
  source,
  error,
  selectedDate,
  modelSlateStatus,
  modelSlateMessage,
}: {
  source: ScheduleSource;
  error: string;
  selectedDate: string;
  modelSlateStatus: ModelSlateStatus;
  modelSlateMessage: string;
}) {
  if (source === 'live' && !error) {
    return (
      <div className="rounded-2xl border border-teal-700/20 bg-teal-700/10 px-4 py-3 text-sm font-semibold text-teal-950 backdrop-blur-md">
        Live MLB schedule loaded for {formatDateLabel(selectedDate)}.{' '}
        {modelSlateStatus === 'loaded'
          ? 'Predictions loaded from your trained model slate.'
          : 'Trained-model slate is missing, so unmatched games use prototype predictions.'}
        {modelSlateStatus !== 'loaded' && modelSlateMessage ? ` ${modelSlateMessage}` : ''}
      </div>
    );
  }

  return (
    <div className="flex items-start gap-3 rounded-2xl border border-amber-500/35 bg-amber-200/55 px-4 py-3 text-sm text-amber-950 backdrop-blur-md">
      <TriangleAlert className="mt-0.5 shrink-0" size={18} />
      <div>
        <strong className="block">Using fallback sample schedule.</strong>
        <span>{error || 'The live schedule API did not return games.'} Predictions use trained slate data where available, otherwise prototype fallback.</span>
      </div>
    </div>
  );
}

function EmptySlate({ selectedDate }: { selectedDate: string }) {
  return (
    <section className="glass-card p-8 text-center">
      <h2 className="text-2xl font-black text-slate-950">No MLB games found</h2>
      <p className="mt-2 text-slate-600">
        MLB&apos;s schedule API returned no games for {formatDateLabel(selectedDate)}. Pick another date to load a different slate.
      </p>
    </section>
  );
}

function DashboardPage({
  predictions,
  selectedGame,
  source,
  error,
  selectedDate,
  modelSlateStatus,
  modelSlateMessage,
  factors,
  factorsAreReal,
  onOpenGame,
}: {
  predictions: GamePrediction[];
  selectedGame?: GamePrediction;
  source: ScheduleSource;
  error: string;
  selectedDate: string;
  modelSlateStatus: ModelSlateStatus;
  modelSlateMessage: string;
  factors: ModelFactor[];
  factorsAreReal: boolean;
  onOpenGame: (game: GamePrediction) => void;
}) {
  if (!predictions.length) return <EmptySlate selectedDate={selectedDate} />;

  const topEdge = Math.max(...predictions.map((game) => Math.abs(game.modelEdge)));
  const averageConfidence =
    predictions.reduce((sum, game) => sum + game.confidenceScore, 0) / predictions.length;
  const powerRankings = buildPowerRankingsFromPredictions(predictions);

  return (
    <div className="space-y-6">
      <SourceBanner source={source} error={error} selectedDate={selectedDate} modelSlateStatus={modelSlateStatus} modelSlateMessage={modelSlateMessage} />

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
        <SummaryMetricCard label="Games modeled today" value={predictions.length} icon={CalendarDays} />
        <SummaryMetricCard label="Highest model edge" value={`${topEdge.toFixed(1)} pts`} icon={BarChart3} tone="positive" />
        <SummaryMetricCard label="Average confidence" value={`${averageConfidence.toFixed(0)}%`} icon={Activity} />
        <SummaryMetricCard label="Model health" value={source === 'live' ? 'Stable' : 'Fallback'} helper={source === 'live' ? 'Live schedule' : 'Sample schedule'} tone={source === 'live' ? 'positive' : 'negative'} />
        <SummaryMetricCard
          label="Prediction layer"
          value={modelSlateStatus === 'loaded' ? 'Trained model' : 'Mixed'}
          helper={modelSlateStatus === 'loaded' ? 'Static slate loaded' : 'Prototype fallback used'}
        />
      </section>

      <section className="grid gap-6 xl:grid-cols-[minmax(0,1.5fr)_minmax(340px,0.8fr)]">
        <div className="space-y-4">
          <div className="flex flex-wrap items-end justify-between gap-3">
            <div>
              <p className="section-kicker">Today&apos;s featured matchups</p>
              <h2 className="section-title">
                Live schedule, {modelSlateStatus === 'loaded' ? 'trained model picks' : 'fallback picks'}
              </h2>
            </div>
            {selectedGame ? (
              <button
                className="rounded-lg border border-slate-300/60 bg-white/55 px-4 py-2 text-sm font-semibold text-slate-800 transition hover:bg-white/75"
                onClick={() => onOpenGame(selectedGame)}
              >
                Open lead matchup
              </button>
            ) : null}
          </div>
          <div className="grid gap-4 lg:grid-cols-2">
            {predictions.slice(0, 4).map((game) => (
              <MatchupCard game={game} key={game.id} onView={onOpenGame} />
            ))}
          </div>
        </div>

        <div className="space-y-4">
          <ModelHealthCard />
          <FactorImpactList
            factors={factors.slice(0, 7)}
            title={factorsAreReal ? 'Model feature importance' : 'Key model factors (prototype)'}
          />
          <PowerRankingsCard teams={powerRankings} />
        </div>
      </section>

      <section className="glass-card p-5">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="section-kicker">Model accuracy snapshot</p>
            <h2 className="section-title">Trailing trend</h2>
          </div>
          <span className="rounded-full border border-slate-300/60 bg-white/55 px-3 py-1 text-sm font-semibold text-slate-700">
            Mock 30-day sample
          </span>
        </div>
        <PerformanceChart data={performanceTrend} />
      </section>
    </div>
  );
}

function TodayGamesPage({
  predictions,
  selectedDate,
  source,
  error,
  modelSlateStatus,
  modelSlateMessage,
  onOpenGame,
}: {
  predictions: GamePrediction[];
  selectedDate: string;
  source: ScheduleSource;
  error: string;
  modelSlateStatus: ModelSlateStatus;
  modelSlateMessage: string;
  onOpenGame: (game: GamePrediction) => void;
}) {
  const [confidence, setConfidence] = useState<ConfidenceFilter>('All');
  const [edgeFilter, setEdgeFilter] = useState<EdgeFilter>('All games');
  const [viewMode, setViewMode] = useState<ViewMode>('probability');

  const games = useMemo(
    () => applyGameFilters(predictions, confidence, edgeFilter),
    [predictions, confidence, edgeFilter],
  );

  if (!predictions.length) return <EmptySlate selectedDate={selectedDate} />;

  return (
    <div className="space-y-5">
      <SourceBanner source={source} error={error} selectedDate={selectedDate} modelSlateStatus={modelSlateStatus} modelSlateMessage={modelSlateMessage} />

      <section className="glass-card p-5">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <p className="section-kicker">Today&apos;s games</p>
            <h2 className="section-title">Slate for {formatDateLabel(selectedDate)}</h2>
            <p className="mt-2 text-sm text-slate-600">
              Schedule is live MLB API data. Prediction fields use your trained model slate when a matching `gamePk`
              is available.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <select className="control" value={confidence} onChange={(event) => setConfidence(event.target.value as ConfidenceFilter)}>
              {confidenceOrder.map((option) => (
                <option key={option}>{option}</option>
              ))}
            </select>
            <select className="control" value={edgeFilter} onChange={(event) => setEdgeFilter(event.target.value as EdgeFilter)}>
              {edgeFilters.map((option) => (
                <option key={option}>{option}</option>
              ))}
            </select>
            <button
              className="control inline-flex items-center gap-2"
              onClick={() => setViewMode(viewMode === 'probability' ? 'score' : 'probability')}
            >
              <SlidersHorizontal size={16} />
              {viewMode === 'probability' ? 'Projected score' : 'Win probability'}
            </button>
          </div>
        </div>
      </section>

      <section className="grid gap-4 xl:grid-cols-3">
        {games.slice(0, 3).map((game) => (
          <MatchupCard game={game} key={game.id} onView={onOpenGame} />
        ))}
      </section>

      <section className="overflow-hidden rounded-2xl border border-slate-300/45 bg-white/55 shadow-sm backdrop-blur-md">
        <div className="grid grid-cols-[88px_1.1fr_1.1fr_1.3fr_1.2fr_105px_90px_1fr_90px_110px] gap-0 border-b border-slate-300/50 px-4 py-3 text-xs font-bold uppercase tracking-[0.14em] text-slate-500 max-xl:hidden">
          <span>Time</span>
          <span>Away</span>
          <span>Home</span>
          <span>Probable pitchers</span>
          <span>{viewMode === 'probability' ? 'Win probability' : 'Projected score'}</span>
          <span>Confidence</span>
          <span>Edge</span>
          <span>Top factor</span>
          <span>Source</span>
          <span>Action</span>
        </div>
        {games.map((game) => (
          <div
            className="grid gap-3 border-b border-slate-300/45 px-4 py-4 text-sm text-slate-700 last:border-b-0 xl:grid-cols-[88px_1.1fr_1.1fr_1.3fr_1.2fr_105px_90px_1fr_90px_110px] xl:items-center"
            key={game.id}
          >
            <span className="font-semibold text-slate-950">{game.time}</span>
            <span className="flex items-center gap-2 font-semibold text-slate-900">
              <TeamIdentity team={game.awayTeam} size="sm" />
              {game.awayTeam.abbreviation}
            </span>
            <span className="flex items-center gap-2 font-semibold text-slate-900">
              <TeamIdentity team={game.homeTeam} size="sm" />
              {game.homeTeam.abbreviation}
            </span>
            <span>{game.awayPitcher} / {game.homePitcher}</span>
            <span>
              {viewMode === 'probability' ? (
                <WinProbabilityBar game={game} compact />
              ) : (
                `${game.projectedAwayRuns.toFixed(1)} - ${game.projectedHomeRuns.toFixed(1)}`
              )}
            </span>
            <span><ConfidenceBadge label={game.confidenceLabel} /></span>
            <span className={game.modelEdge >= 0 ? 'font-bold text-teal-800' : 'font-bold text-red-800'}>
              {game.modelEdge > 0 ? '+' : ''}{game.modelEdge.toFixed(1)}
            </span>
            <span>{game.topFactors[0]?.name}</span>
            <span className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">
              {game.predictionSource === 'model' ? 'Model' : 'Fallback'}
            </span>
            <button className="rounded-lg bg-teal-700/12 px-3 py-2 font-semibold text-teal-900 hover:bg-teal-700/20" onClick={() => onOpenGame(game)}>
              Analyze
            </button>
          </div>
        ))}
      </section>
    </div>
  );
}

function ModelAccuracyPage({
  selectedDate,
}: {
  selectedDate: string;
}) {
  const [window, setWindow] = useState<AccuracyWindow>('30 days');
  const [records, setRecords] = useState(accuracyRecords);
  const [historyError, setHistoryError] = useState('');
  const [loadingHistory, setLoadingHistory] = useState(true);
  const days = window === '7 days' ? 7 : window === '30 days' ? 30 : 90;

  useEffect(() => {
    let active = true;
    setLoadingHistory(true);
    setHistoryError('');

    fetchMlbScheduleRange(shiftDateString(selectedDate, -days), selectedDate)
      .then((games) => {
        if (!active) return;
        const realRecords = games
          .filter((game) => game.isFinal)
          .map(generateAccuracyRecordForGame)
          .filter((record): record is NonNullable<typeof record> => record !== null)
          .sort((a, b) => b.date.localeCompare(a.date))
          .slice(0, window === 'Season to date' ? 80 : 40);
        setRecords(realRecords);
      })
      .catch((error: unknown) => {
        if (!active) return;
        setHistoryError(error instanceof Error ? error.message : 'Unable to load recent MLB results.');
        setRecords(accuracyRecords);
      })
      .finally(() => {
        if (active) setLoadingHistory(false);
      });

    return () => {
      active = false;
    };
  }, [selectedDate, days, window]);

  return (
    <div className="space-y-5">
      <section className="glass-card flex flex-wrap items-center justify-between gap-4 p-5">
        <div>
          <p className="section-kicker">Model accuracy</p>
          <h2 className="section-title">Trailing 30-day performance</h2>
          <p className="mt-2 text-sm text-slate-600">
            Games and final scores are loaded from MLB schedule results. Historical predicted winners use saved model
            history when connected; this view currently rebuilds comparable predictions from the available slate layer.
          </p>
          {historyError ? (
            <p className="mt-2 text-sm font-semibold text-amber-900">
              Recent results failed to load, so this table is using fallback sample history: {historyError}
            </p>
          ) : null}
        </div>
        <div className="flex rounded-xl border border-slate-300/60 bg-white/55 p-1">
          {(['7 days', '30 days', 'Season to date'] as AccuracyWindow[]).map((option) => (
            <button
              className={`rounded-lg px-3 py-2 text-sm font-semibold transition ${
                window === option ? 'bg-slate-900 text-white' : 'text-slate-700 hover:bg-white/70'
              }`}
              key={option}
              onClick={() => setWindow(option)}
            >
              {option}
            </button>
          ))}
        </div>
      </section>
      {loadingHistory ? (
        <section className="glass-card p-5 text-sm font-semibold text-slate-700">
          Loading recent MLB results...
        </section>
      ) : (
        <AccuracyOverview records={records} />
      )}
      <section className="grid gap-5 xl:grid-cols-2">
        <CalibrationChart data={calibrationData} />
        <div className="glass-card p-5">
          <h3 className="mb-4 text-lg font-semibold text-slate-950">Daily accuracy and ROI</h3>
          <PerformanceChart data={performanceTrend} compact />
        </div>
      </section>
      <RecentPredictionsTable records={records} />
    </div>
  );
}

function ModelFactorsPage({
  factors,
  model,
  factorsAreReal,
}: {
  factors: ModelFactor[];
  model?: SlateModelSummary;
  factorsAreReal: boolean;
}) {
  return (
    <div className="space-y-5">
      <section className="glass-card p-5">
        <p className="section-kicker">Model factors</p>
        <h2 className="section-title">How Diamond Forecast evaluates a game</h2>
        {factorsAreReal ? (
          <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-600">
            Feature groups below are the deployed <strong>{model?.mode ?? 'pregame_safe'}</strong> model&apos;s
            real importances ({model?.featureCount ?? '—'} features, {model?.importanceMetric ?? 'split gain'}),
            grouped by data source. Percentages are each group&apos;s share of total model gain. Per-game signed
            attributions are a planned SHAP extension.
          </p>
        ) : (
          <p className="mt-3 max-w-3xl text-sm leading-6 text-amber-900">
            No trained-model slate is loaded for this date, so these are prototype placeholder factors. Build a
            slate with <code>python scripts/build_static_slate.py</code> to see real model importances.
          </p>
        )}
      </section>
      <div className="grid gap-4 lg:grid-cols-2">
        {factors.map((factor) => (
          <article className="glass-card p-5" key={factor.name}>
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-xs font-bold uppercase tracking-[0.18em] text-slate-500">{factor.category}</p>
                <h3 className="mt-2 text-lg font-semibold text-slate-950">{factor.name}</h3>
              </div>
              {factorsAreReal ? (
                <span className="rounded-full bg-teal-700/12 px-3 py-1 text-xs font-black text-teal-900">
                  {factor.impact}% gain
                </span>
              ) : (
                <span className={`rounded-full px-3 py-1 text-xs font-bold ${factor.direction === 'Positive' ? 'bg-teal-700/12 text-teal-900' : factor.direction === 'Negative' ? 'bg-red-500/12 text-red-900' : 'bg-slate-500/12 text-slate-700'}`}>
                  {factor.direction}
                </span>
              )}
            </div>
            <p className="mt-3 text-sm leading-6 text-slate-600">{factor.description}</p>
            <div className="mt-4 rounded-xl border border-slate-300/50 bg-white/45 p-3 text-sm text-slate-700">
              <strong className="text-slate-950">Example signal:</strong> {factor.exampleSignal}
            </div>
            <div className="mt-4 flex flex-wrap gap-2">
              {factor.affects.map((item) => (
                <span className="rounded-full border border-slate-300/60 bg-white/35 px-3 py-1 text-xs font-semibold text-slate-700" key={item}>
                  {item}
                </span>
              ))}
            </div>
          </article>
        ))}
      </div>
    </div>
  );
}

function SettingsPage({ source }: { source: ScheduleSource }) {
  return (
    <div className="grid gap-5 xl:grid-cols-[1fr_0.8fr]">
      <section className="glass-card p-5">
        <p className="section-kicker">Settings</p>
        <h2 className="section-title">Prototype controls</h2>
        <div className="mt-5 grid gap-4 md:grid-cols-2">
          {['Refresh schedule automatically', 'Show market line deltas', 'Use compact table density', 'Highlight injury risk'].map((label, index) => (
            <label className="flex items-center justify-between rounded-xl border border-slate-300/50 bg-white/45 p-4 text-sm font-semibold text-slate-800" key={label}>
              {label}
              <input className="h-5 w-5 accent-teal-700" type="checkbox" defaultChecked={index < 2} />
            </label>
          ))}
        </div>
      </section>
      <section className="glass-card p-5">
        <h3 className="text-lg font-semibold text-slate-950">Data sources</h3>
        <p className="mt-3 text-sm leading-6 text-slate-600">
          Schedule source: {source === 'live' ? 'MLB Stats API live schedule' : 'fallback sample schedule'}. Prediction
          fields use trained slate data when available, with prototype fallback only for dates missing a model slate.
        </p>
      </section>
    </div>
  );
}

function BacktestPage() {
  return (
    <div className="space-y-5">
      <section className="glass-card p-5">
        <p className="section-kicker">Backtest</p>
        <h2 className="section-title">Historical validation snapshot</h2>
        <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-600">
          Mock backtest view for held-out season metrics, calibration checks, and edge-tracking return summaries.
        </p>
      </section>
      <AccuracyOverview records={accuracyRecords} />
      <RecentPredictionsTable records={accuracyRecords.slice(0, 8)} />
    </div>
  );
}

function App() {
  const [activeNav, setActiveNav] = useState<NavKey>('dashboard');
  const [selectedDate, setSelectedDate] = useState(browserToday);
  const [scheduleGames, setScheduleGames] = useState<MlbGame[]>([]);
  const [scheduleSource, setScheduleSource] = useState<ScheduleSource>('live');
  const [scheduleError, setScheduleError] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState('Loading');
  const [selectedGameId, setSelectedGameId] = useState<string>('');
  const [modelSlateStatus, setModelSlateStatus] = useState<ModelSlateStatus>('missing');
  const [modelSlateMessage, setModelSlateMessage] = useState('');
  const [modelSummary, setModelSummary] = useState<SlateModelSummary | undefined>(undefined);
  const [slateProvenance, setSlateProvenance] = useState<SlateProvenance | undefined>(undefined);
  const [modelPredictionsByGamePk, setModelPredictionsByGamePk] = useState<Map<number, ReturnType<typeof buildPredictionFromModelSlate>>>(new Map());

  useEffect(() => {
    let active = true;
    setIsLoading(true);
    setScheduleError('');

    fetchMlbSchedule(selectedDate)
      .then((games) => {
        if (!active) return;
        setScheduleGames(games);
        setScheduleSource('live');
        setLastUpdated(lastUpdatedLabel());
        setSelectedGameId((current) => (games.some((game) => game.id === current) ? current : games[0]?.id ?? ''));
      })
      .catch((error: unknown) => {
        if (!active) return;
        const message = error instanceof Error ? error.message : 'Unable to load MLB schedule.';
        setScheduleError(message);
        setScheduleGames(
          fallbackScheduleGames.map((game) => ({
            ...game,
            date: selectedDate,
          })),
        );
        setScheduleSource('fallback');
        setLastUpdated(lastUpdatedLabel());
        setSelectedGameId(fallbackScheduleGames[0]?.id ?? '');
      })
      .finally(() => {
        if (active) setIsLoading(false);
      });

    return () => {
      active = false;
    };
  }, [selectedDate]);

  useEffect(() => {
    let active = true;
    setModelSlateMessage('');
    setModelSlateStatus('missing');
    setModelSummary(undefined);
    setSlateProvenance(undefined);
    setModelPredictionsByGamePk(new Map());

    fetchModelSlate(selectedDate).then((result) => {
      if (!active) return;
      setModelSlateStatus(result.status);
      setModelSlateMessage(result.message ?? '');
      setModelSummary(result.model);
      setSlateProvenance(result.provenance);
      setModelPredictionsByGamePk(() => {
        const next = new Map<number, ReturnType<typeof buildPredictionFromModelSlate>>();
        scheduleGames.forEach((game) => {
          const staticGame = result.predictionsByGamePk.get(game.gamePk);
          if (!staticGame) return;
          next.set(game.gamePk, buildPredictionFromModelSlate(game, staticGame));
        });
        return next;
      });
    });

    return () => {
      active = false;
    };
  }, [selectedDate, scheduleGames]);

  const predictions = useMemo(
    () =>
      scheduleGames.map((game) => {
        const modelPrediction = modelPredictionsByGamePk.get(game.gamePk);
        return modelPrediction ?? generateMockPredictionForGame(game);
      }),
    [scheduleGames, modelPredictionsByGamePk],
  );
  const selectedGame = predictions.find((game) => game.id === selectedGameId) ?? predictions[0];

  // Real LightGBM importances from the loaded slate; mock factors only when no
  // trained-model slate is available for the date.
  const factorsAreReal = Boolean(modelSummary?.factorGroups?.length);
  const modelFactorList = useMemo<ModelFactor[]>(
    () => (factorsAreReal ? modelSummaryToFactors(modelSummary) : modelFactors),
    [factorsAreReal, modelSummary],
  );
  const updatedLabel = slateProvenance?.dataAsOf
    ? formatAsOfLabel(slateProvenance.dataAsOf)
    : lastUpdated;

  const openGame = (game: GamePrediction) => {
    setSelectedGameId(game.id);
    setActiveNav('game-detail');
  };

  const content = {
    dashboard: (
      <DashboardPage
        predictions={predictions}
        selectedGame={selectedGame}
        source={scheduleSource}
        error={scheduleError}
        selectedDate={selectedDate}
        modelSlateStatus={modelSlateStatus}
        modelSlateMessage={modelSlateMessage}
        factors={modelFactorList}
        factorsAreReal={factorsAreReal}
        onOpenGame={openGame}
      />
    ),
    games: (
      <TodayGamesPage
        predictions={predictions}
        selectedDate={selectedDate}
        source={scheduleSource}
        error={scheduleError}
        modelSlateStatus={modelSlateStatus}
        modelSlateMessage={modelSlateMessage}
        onOpenGame={openGame}
      />
    ),
    'game-detail': selectedGame ? <GameDetailPanel game={selectedGame} /> : <EmptySlate selectedDate={selectedDate} />,
    accuracy: <ModelAccuracyPage selectedDate={selectedDate} />,
    factors: <ModelFactorsPage factors={modelFactorList} model={modelSummary} factorsAreReal={factorsAreReal} />,
    backtest: <BacktestPage />,
    settings: <SettingsPage source={scheduleSource} />,
  } satisfies Record<NavKey, ReactElement>;

  return (
    <div className="min-h-screen bg-slate-100 text-slate-950">
      <div className="stadium-bg fixed inset-0" aria-hidden="true" />
      <div className="fixed inset-0 bg-white/58 backdrop-blur-[1px]" aria-hidden="true" />
      <div className="relative z-10 flex min-h-screen">
        <Sidebar active={activeNav} onNavigate={setActiveNav} />
        <main className="min-w-0 flex-1 px-4 py-4 md:px-6 lg:px-8">
          <TopNav
            selectedDate={selectedDate}
            onDateChange={setSelectedDate}
            onStepDate={(days) => setSelectedDate((date) => shiftDate(date, days))}
            lastUpdated={updatedLabel}
            sourceLabel={scheduleSource === 'live' ? 'Live schedule' : 'Fallback data'}
            usingFallback={scheduleSource === 'fallback'}
          />
          <ProvenanceBar
            provenance={slateProvenance}
            modelSlateStatus={modelSlateStatus}
            selectedDate={selectedDate}
          />
          <nav className="no-scrollbar mt-4 flex gap-2 overflow-x-auto rounded-2xl border border-slate-300/45 bg-white/55 p-2 shadow-sm backdrop-blur-md lg:hidden" aria-label="Dashboard sections">
            {compactNavItems.map((item) => (
              <button
                className={`whitespace-nowrap rounded-xl px-3 py-2 text-sm font-semibold transition ${
                  activeNav === item.key ? 'bg-slate-900 text-white' : 'text-slate-700 hover:bg-white/75'
                }`}
                key={item.key}
                onClick={() => setActiveNav(item.key)}
              >
                {item.label}
              </button>
            ))}
          </nav>
          <div className="mt-6">
            {isLoading ? (
              <section className="glass-card flex items-center gap-3 p-6 text-slate-700">
                <RefreshCw className="animate-spin" size={20} />
                Loading MLB schedule for {formatDateLabel(selectedDate)}...
              </section>
            ) : (
              content[activeNav]
            )}
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
