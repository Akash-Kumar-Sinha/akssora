"use client";

import React, { useMemo, useRef, useState, useEffect } from "react";
import { cn } from "@/lib/utils";
import { Play, Pause, Volume2, VolumeX, List } from "lucide-react";

export interface Chapter {
  time: number;
  label?: string;
}

interface MediaCardProps {
  url: string;
  className?: string;
  chapters?: Chapter[];
}

export const MediaCard = ({
  url,
  className,
  chapters = [],
}: MediaCardProps) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isHovered, setIsHovered] = useState(false);
  const [activeChapter, setActiveChapter] = useState<number>(-1);
  const [isMuted, setIsMuted] = useState(true);
  const [showLabels, setShowLabels] = useState(false);

  const isVideo = useMemo(() => {
    if (!url) return false;
    const lowerUrl = url.toLowerCase();
    return (
      lowerUrl.endsWith(".mp4") ||
      lowerUrl.endsWith(".webm") ||
      lowerUrl.endsWith(".ogg") ||
      lowerUrl.includes("video")
    );
  }, [url]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleTimeUpdate = () => {
      const t = video.currentTime;
      setProgress((t / video.duration) * 100);
      setCurrentTime(t);

      if (chapters.length > 0) {
        let idx = -1;
        for (let i = chapters.length - 1; i >= 0; i--) {
          if (chapters[i].time && t >= chapters[i].time) {
            idx = i;
            break;
          }
        }
        setActiveChapter(idx);
      }
    };

    const handleLoadedMetadata = () => setDuration(video.duration);

    const handleEnded = () => {
      setIsPlaying(false);
      setProgress(100);
      setCurrentTime(video.duration);
    };

    video.addEventListener("timeupdate", handleTimeUpdate);
    video.addEventListener("loadedmetadata", handleLoadedMetadata);
    video.addEventListener("ended", handleEnded);

    return () => {
      video.removeEventListener("timeupdate", handleTimeUpdate);
      video.removeEventListener("loadedmetadata", handleLoadedMetadata);
      video.removeEventListener("ended", handleEnded);
    };
  }, [chapters]);

  const togglePlay = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (videoRef.current) {
      isPlaying ? videoRef.current.pause() : videoRef.current.play();
      setIsPlaying(!isPlaying);
    }
  };

  const toggleMute = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (videoRef.current) {
      videoRef.current.muted = !isMuted;
      setIsMuted(!isMuted);
    }
  };

  const handleSeek = (e: React.MouseEvent<HTMLDivElement>) => {
    e.stopPropagation();
    if (videoRef.current) {
      const bounds = e.currentTarget.getBoundingClientRect();
      const percent = Math.max(
        0,
        Math.min(1, (e.clientX - bounds.left) / bounds.width),
      );
      const seekTime = percent * videoRef.current.duration;
      videoRef.current.currentTime = seekTime;
      setProgress(percent * 100);
      setCurrentTime(seekTime);
    }
  };

  const jumpToChapter = (time: number, index: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = time;
      setCurrentTime(time);
      setProgress((time / videoRef.current.duration) * 100);
      setActiveChapter(index);
      if (!isPlaying) {
        videoRef.current.play();
        setIsPlaying(true);
      }
    }
  };

  const formatTime = (seconds: number) => {
    if (isNaN(seconds) || !isFinite(seconds)) return "0:00";
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  return (
    <div
      className={cn(
        "overflow-hidden rounded-4xl bg-zinc-950 backdrop-blur-xl border border-white/20 shadow-[0_8px_32px_rgba(0,0,0,0.4),inset_0_1px_0_rgba(255,255,255,0.1)] transition-all duration-500 hover:scale-[1.02] hover:shadow-2xl hover:shadow-black/30",
        className,
      )}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {isVideo ? (
        <>
          {/* ── Video player ── */}
          <div
            className="relative cursor-pointer group aspect-video bg-black"
            onClick={togglePlay}
          >
            <video
              ref={videoRef}
              src={url}
              loop
              muted
              playsInline
              className="w-full h-full object-contain"
            />

            {/* Controls overlay */}
            <div
              className={cn(
                "absolute inset-0 flex flex-col justify-end bg-linear-to-t from-black/80 via-black/10 to-transparent transition-opacity duration-300",
                isHovered || !isPlaying ? "opacity-100" : "opacity-0",
              )}
            >
              {/* No centre button — clicking the video area handles play/pause */}

              {/* Bottom controls */}
              <div
                className="p-2 flex flex-col gap-2"
                onClick={(e) => e.stopPropagation()}
              >
                <div
                  className="w-full h-1 bg-white/30 rounded-full overflow-hidden cursor-pointer hover:h-1.5 transition-all"
                  onClick={handleSeek}
                >
                  <div
                    className="h-full bg-white rounded-full transition-all duration-100 ease-linear"
                    style={{ width: `${progress}%` }}
                  />
                </div>

                <div className="flex items-center gap-4 text-white">
                  <button
                    onClick={togglePlay}
                    className="focus:outline-none hover:opacity-70 transition-opacity"
                  >
                    {isPlaying ? (
                      <Pause className="w-5 h-5" strokeWidth={2.5} />
                    ) : (
                      <Play
                        className="w-5 h-5"
                        strokeWidth={2.5}
                        fill="currentColor"
                      />
                    )}
                  </button>
                  <span className="text-xs font-mono tracking-wider opacity-80">
                    {formatTime(currentTime)} / {formatTime(duration)}
                  </span>

                  <button
                    onClick={toggleMute}
                    className="ml-auto focus:outline-none hover:opacity-70 transition-opacity"
                  >
                    {isMuted ? (
                      <VolumeX className="w-5 h-5" />
                    ) : (
                      <Volume2 className="w-5 h-5" />
                    )}
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* ── Chapter timeline ── */}
          {chapters.length > 0 && (
            <div>
              {/* Header row with toggle */}
              <div className="flex items-center justify-between px-4 py-2 border-t border-white/20">
                <span className="text-[10px] font-mono tracking-widest text-white/60 uppercase">
                  Chapters
                </span>
                <button
                  onClick={() => setShowLabels((v) => !v)}
                  className={cn(
                    "flex items-center gap-1 text-[10px] font-mono tracking-wider px-2 py-0.5 rounded-full border transition-colors duration-200",
                    showLabels
                      ? "border-white/40 text-white bg-white/15"
                      : "border-white/20 text-white/60 hover:border-white/30 hover:text-white/90",
                  )}
                >
                  <List className="w-3 h-3" />
                  {showLabels ? "Hide labels" : "Show labels"}
                </button>
              </div>

              {showLabels ? (
                /* ── Full label list ── */
                <div className="custom-scroll max-h-48 sm:max-h-56 md:max-h-64 lg:max-h-80 xl:max-h-96 overflow-y-auto">
                  {chapters.map((chapter, i) => {
                    const isActive = i === activeChapter;
                    return (
                      <button
                        key={i}
                        onClick={() => jumpToChapter(chapter.time, i)}
                        className={cn(
                          "w-full flex items-start gap-3 px-4 py-3 text-left transition-colors duration-200",
                          isActive
                            ? "bg-white/15 text-white"
                            : "text-white/70 hover:bg-white/8 hover:text-white",
                          i !== 0 && "border-t border-white/10",
                        )}
                      >
                        <div className="flex flex-col gap-0.5 min-w-0">
                          <span className="text-[10px] font-mono tracking-widest text-white/50">
                            {formatTime(chapter.time)}
                          </span>
                          <span className="text-xs font-semibold leading-snug break-words">
                            {chapter.label}
                          </span>
                        </div>
                      </button>
                    );
                  })}
                </div>
              ) : (
                /* ── Compact badge row ── */
                <div className="flex flex-wrap gap-2 px-4 py-3">
                  {chapters.map((chapter, i) => {
                    const isActive = i === activeChapter;
                    return (
                      <button
                        key={i}
                        onClick={() => jumpToChapter(chapter.time, i)}
                        title={chapter.label}
                        className={cn(
                          "text-[10px] font-mono tracking-wider px-2.5 py-1 rounded-full border transition-colors duration-200",
                          isActive
                            ? "bg-white/25 border-white/50 text-white"
                            : "bg-white/8 border-white/20 text-white/70 hover:bg-white/15 hover:border-white/35 hover:text-white",
                        )}
                      >
                        {formatTime(chapter.time)}
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          )}
        </>
      ) : (
        /* ── Image ── */
        <div className="relative aspect-video bg-black">
          <img
            src={url}
            alt="Media content"
            className="absolute inset-0 w-full h-full object-cover"
          />
        </div>
      )}
    </div>
  );
};
