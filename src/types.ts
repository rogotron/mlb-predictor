export type PaletteColor = 'red' | 'blue' | 'green' | 'yellow' | 'black' | 'white' | 'orange';

export type RegionShape =
  | {
      kind: 'rect';
      x: number;
      y: number;
      width: number;
      height: number;
    }
  | {
      kind: 'circle';
      cx: number;
      cy: number;
      r: number;
    };

export interface FlagRegion {
  id: string;
  label: string;
  correctColor: PaletteColor;
  shape: RegionShape;
}

export interface FlagData {
  id: string;
  country: string;
  viewBox: string;
  regions: FlagRegion[];
}

export type RegionStatus = 'correct' | 'incorrect';
