import { colorHex } from '../data/flags';
import type { FlagData, PaletteColor, RegionStatus } from '../types';

interface FlagCanvasProps {
  flag: FlagData;
  paintedRegions: Record<string, PaletteColor>;
  results: Record<string, RegionStatus>;
  onPaintRegion: (regionId: string) => void;
}

export function FlagCanvas({ flag, paintedRegions, results, onPaintRegion }: FlagCanvasProps) {
  return (
    <div className="flag-card">
      <svg aria-label={`Blank flag for ${flag.country}`} className="flag-svg" role="img" viewBox={flag.viewBox}>
        {flag.regions.map((region) => {
          const fillColor = paintedRegions[region.id] ? colorHex[paintedRegions[region.id]] : '#ffffff';
          const status = results[region.id];
          const className = `flag-region ${status ?? ''}`;

          if (region.shape.kind === 'circle') {
            return (
              <circle
                aria-label={region.label}
                className={className}
                cx={region.shape.cx}
                cy={region.shape.cy}
                fill={fillColor}
                key={region.id}
                onClick={() => onPaintRegion(region.id)}
                r={region.shape.r}
                role="button"
                tabIndex={0}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' || event.key === ' ') {
                    onPaintRegion(region.id);
                  }
                }}
              />
            );
          }

          return (
            <rect
              aria-label={region.label}
              className={className}
              fill={fillColor}
              height={region.shape.height}
              key={region.id}
              onClick={() => onPaintRegion(region.id)}
              role="button"
              tabIndex={0}
              width={region.shape.width}
              x={region.shape.x}
              y={region.shape.y}
              onKeyDown={(event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                  onPaintRegion(region.id);
                }
              }}
            />
          );
        })}
      </svg>
    </div>
  );
}
