import { useState } from 'react';

type DisplayTeam = {
  name: string;
  abbreviation: string;
  logoUrl?: string;
  helmetUrl?: string;
};

export function TeamIdentity({ team, size = 'md' }: { team: DisplayTeam; size?: 'sm' | 'md' | 'lg' }) {
  const [failed, setFailed] = useState(false);
  const sizeClass = size === 'lg' ? 'h-14 w-14 text-lg' : size === 'sm' ? 'h-9 w-9 text-xs' : 'h-11 w-11 text-sm';
  const imageUrl = team.helmetUrl ?? team.logoUrl;

  return (
    <div
      className={`${sizeClass} relative grid shrink-0 place-items-center overflow-hidden rounded-full border border-slate-200/80 bg-white/70 font-black text-slate-900 shadow-sm`}
      title={team.name}
    >
      {failed || !imageUrl ? (
        <span>{team.abbreviation}</span>
      ) : (
        <img
          alt=""
          className="h-full w-full object-contain p-1.5"
          src={imageUrl}
          onError={() => setFailed(true)}
        />
      )}
    </div>
  );
}
