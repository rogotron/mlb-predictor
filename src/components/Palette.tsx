import { colorHex, paletteColors } from '../data/flags';
import type { PaletteColor } from '../types';

interface PaletteProps {
  selectedColor: PaletteColor;
  onSelectColor: (color: PaletteColor) => void;
}

export function Palette({ selectedColor, onSelectColor }: PaletteProps) {
  return (
    <section className="palette-panel" aria-label="Color palette">
      <h2>Paint colors</h2>
      <div className="palette-grid">
        {paletteColors.map((color) => (
          <button
            aria-pressed={selectedColor === color.name}
            className={`color-swatch ${selectedColor === color.name ? 'selected' : ''}`}
            key={color.name}
            onClick={() => onSelectColor(color.name)}
            style={{ '--swatch': colorHex[color.name] } as React.CSSProperties}
            type="button"
          >
            <span className="swatch-dot" />
            <span>{color.name}</span>
          </button>
        ))}
      </div>
    </section>
  );
}
