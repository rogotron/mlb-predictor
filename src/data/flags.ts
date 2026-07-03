import type { FlagData, PaletteColor } from '../types';

export const paletteColors: Array<{ name: PaletteColor; hex: string }> = [
  { name: 'red', hex: '#e3342f' },
  { name: 'blue', hex: '#2563eb' },
  { name: 'green', hex: '#16a34a' },
  { name: 'yellow', hex: '#facc15' },
  { name: 'black', hex: '#111827' },
  { name: 'white', hex: '#ffffff' },
  { name: 'orange', hex: '#f97316' },
];

export const colorHex = Object.fromEntries(
  paletteColors.map((color) => [color.name, color.hex]),
) as Record<PaletteColor, string>;

const thirds = [
  { x: 0, y: 0, width: 100, height: 200 },
  { x: 100, y: 0, width: 100, height: 200 },
  { x: 200, y: 0, width: 100, height: 200 },
];

const horizontalThirds = [
  { x: 0, y: 0, width: 300, height: 66.67 },
  { x: 0, y: 66.67, width: 300, height: 66.66 },
  { x: 0, y: 133.33, width: 300, height: 66.67 },
];

const halves = [
  { x: 0, y: 0, width: 300, height: 100 },
  { x: 0, y: 100, width: 300, height: 100 },
];

export const flags: FlagData[] = [
  {
    id: 'france',
    country: 'France',
    viewBox: '0 0 300 200',
    regions: [
      { id: 'blue-stripe', label: 'Left stripe', correctColor: 'blue', shape: { kind: 'rect', ...thirds[0] } },
      { id: 'white-stripe', label: 'Middle stripe', correctColor: 'white', shape: { kind: 'rect', ...thirds[1] } },
      { id: 'red-stripe', label: 'Right stripe', correctColor: 'red', shape: { kind: 'rect', ...thirds[2] } },
    ],
  },
  {
    id: 'italy',
    country: 'Italy',
    viewBox: '0 0 300 200',
    regions: [
      { id: 'green-stripe', label: 'Left stripe', correctColor: 'green', shape: { kind: 'rect', ...thirds[0] } },
      { id: 'white-stripe', label: 'Middle stripe', correctColor: 'white', shape: { kind: 'rect', ...thirds[1] } },
      { id: 'red-stripe', label: 'Right stripe', correctColor: 'red', shape: { kind: 'rect', ...thirds[2] } },
    ],
  },
  {
    id: 'germany',
    country: 'Germany',
    viewBox: '0 0 300 200',
    regions: [
      { id: 'black-band', label: 'Top band', correctColor: 'black', shape: { kind: 'rect', ...horizontalThirds[0] } },
      { id: 'red-band', label: 'Middle band', correctColor: 'red', shape: { kind: 'rect', ...horizontalThirds[1] } },
      { id: 'yellow-band', label: 'Bottom band', correctColor: 'yellow', shape: { kind: 'rect', ...horizontalThirds[2] } },
    ],
  },
  {
    id: 'netherlands',
    country: 'Netherlands',
    viewBox: '0 0 300 200',
    regions: [
      { id: 'red-band', label: 'Top band', correctColor: 'red', shape: { kind: 'rect', ...horizontalThirds[0] } },
      { id: 'white-band', label: 'Middle band', correctColor: 'white', shape: { kind: 'rect', ...horizontalThirds[1] } },
      { id: 'blue-band', label: 'Bottom band', correctColor: 'blue', shape: { kind: 'rect', ...horizontalThirds[2] } },
    ],
  },
  {
    id: 'ukraine',
    country: 'Ukraine',
    viewBox: '0 0 300 200',
    regions: [
      { id: 'blue-half', label: 'Top half', correctColor: 'blue', shape: { kind: 'rect', ...halves[0] } },
      { id: 'yellow-half', label: 'Bottom half', correctColor: 'yellow', shape: { kind: 'rect', ...halves[1] } },
    ],
  },
  {
    id: 'poland',
    country: 'Poland',
    viewBox: '0 0 300 200',
    regions: [
      { id: 'white-half', label: 'Top half', correctColor: 'white', shape: { kind: 'rect', ...halves[0] } },
      { id: 'red-half', label: 'Bottom half', correctColor: 'red', shape: { kind: 'rect', ...halves[1] } },
    ],
  },
  {
    id: 'indonesia',
    country: 'Indonesia',
    viewBox: '0 0 300 200',
    regions: [
      { id: 'red-half', label: 'Top half', correctColor: 'red', shape: { kind: 'rect', ...halves[0] } },
      { id: 'white-half', label: 'Bottom half', correctColor: 'white', shape: { kind: 'rect', ...halves[1] } },
    ],
  },
  {
    id: 'belgium',
    country: 'Belgium',
    viewBox: '0 0 300 200',
    regions: [
      { id: 'black-stripe', label: 'Left stripe', correctColor: 'black', shape: { kind: 'rect', ...thirds[0] } },
      { id: 'yellow-stripe', label: 'Middle stripe', correctColor: 'yellow', shape: { kind: 'rect', ...thirds[1] } },
      { id: 'red-stripe', label: 'Right stripe', correctColor: 'red', shape: { kind: 'rect', ...thirds[2] } },
    ],
  },
  {
    id: 'japan',
    country: 'Japan',
    viewBox: '0 0 300 200',
    regions: [
      { id: 'white-field', label: 'Background', correctColor: 'white', shape: { kind: 'rect', x: 0, y: 0, width: 300, height: 200 } },
      { id: 'red-sun', label: 'Circle', correctColor: 'red', shape: { kind: 'circle', cx: 150, cy: 100, r: 48 } },
    ],
  },
  {
    id: 'bangladesh',
    country: 'Bangladesh',
    viewBox: '0 0 300 200',
    regions: [
      { id: 'green-field', label: 'Background', correctColor: 'green', shape: { kind: 'rect', x: 0, y: 0, width: 300, height: 200 } },
      { id: 'red-disc', label: 'Circle', correctColor: 'red', shape: { kind: 'circle', cx: 135, cy: 100, r: 52 } },
    ],
  },
];
