"use client";

import React from "react";
import Upload from "@/components/upload";
import { MediaCard } from "@/components/ui/media-card";
import api from "@/lib/api";
import { BACKEND_URL } from "@/lib/constant";

type Segment = {
  end_time: number;
  score: number;
  start_time: number;
  transcript: string;
};

type SearchResult = {
  score: number;
  type: "video" | "image" | string;
  url: string | null;
  segments?: Segment[];
  transcript?: string;
};

const segmentsToChapters = (segments?: Segment[]) => {
  if (!segments) return [];
  return segments
    .filter((s) => s.transcript?.trim())
    .map((s) => ({
      time: s.start_time,
      label: s.transcript,
    }));
};

const Workspace = () => {
  const [searchQuery, setSearchQuery] = React.useState("what to learn");
  const [results, setResults] = React.useState<SearchResult[]>([]);
  const [loading, setLoading] = React.useState(false);
  const [searched, setSearched] = React.useState(false);

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;

    setLoading(true);
    setSearched(true);

    try {
      const res = await api.get(`${BACKEND_URL}/app/search`, {
        params: { searchQuery },
      });
      setResults(res.data.results ?? []);
    } catch (e) {
      console.error(e);
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") handleSearch();
  };

  return (
    <div className="flex flex-col h-screen bg-black ">
      <main className="flex-1 overflow-y-auto px-6 md:px-10 lg:px-16 pt-20 pb-4">
        <div className="flex gap-2 mb-8 max-w-3xl">
          <input
            type="text"
            value={searchQuery}
            placeholder="Search across your media..."
            className="flex-1 bg-zinc-900 border border-zinc-800 rounded-lg text-white placeholder:text-zinc-500 px-4 py-3 text-sm font-mono outline-none focus:border-zinc-600 transition-colors"
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={handleKeyDown}
          />
          <button
            onClick={handleSearch}
            disabled={loading}
            className="bg-indigo-600 hover:bg-indigo-700 disabled:bg-zinc-800 text-white border-none rounded-lg px-5 py-3 text-xs cursor-pointer disabled:cursor-not-allowed whitespace-nowrap tracking-wide transition-colors"
          >
            {loading ? "searching..." : "search →"}
          </button>
        </div>

        {/* Results */}
        {searched && (
          <div>
            {loading ? (
              <p className="text-zinc-500 text-xs animate-pulse">
                searching...
              </p>
            ) : results.length === 0 ? (
              <p className="text-zinc-600 text-xs">No results found.</p>
            ) : (
              <div>
                <p className="text-zinc-500 text-[11px] mb-5 tracking-wide">
                  {results.length} result{results.length !== 1 ? "s" : ""}
                </p>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 items-start">
                  {results.map((result, i) => (
                    <div key={i} className="flex flex-col">
                      <MediaCard
                        url={result.url ?? ""}
                        chapters={segmentsToChapters(result.segments)}
                        className="w-full"
                      />

                      {result.type === "image" && result.transcript && (
                        <p className="mt-2.5 text-zinc-400 text-xs leading-relaxed">
                          {result.transcript}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {!searched && (
          <div className="flex flex-col items-center justify-center py-20 text-center">
            <p className="text-zinc-600 text-sm tracking-wide">
              Search your media to get started
            </p>
          </div>
        )}
      </main>

      <div>
      </div>
    </div>
  );
};

export default Workspace;
