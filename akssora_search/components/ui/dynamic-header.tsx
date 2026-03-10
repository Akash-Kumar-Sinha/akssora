"use client";

import { useState } from "react";
import Link from "next/link";
import {
  motion,
  AnimatePresence,
  useScroll,
  useTransform,
  useMotionTemplate,
} from "motion/react";
import { cn } from "@/lib/utils";
import { ArrowUpRight, Menu, X } from "lucide-react";
import { Logo } from "@/components/Logo/Logo";

interface NavItem {
  title: string;
  href: string;
  image: string;
  description: string;
}

export const DynamicHeader = ({ nav_items }: { nav_items: NavItem[] }) => {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [isMobileOpen, setIsMobileOpen] = useState(false);

  const { scrollY } = useScroll();
  const bgOpacity = useTransform(scrollY, [0, 60], [0, 0.9]);
  const blurValue = useTransform(scrollY, [0, 60], [0, 16]);
  const padValue = useTransform(scrollY, [0, 60], [12, 4]); // Shrinks padding from 12px to 4px

  const backgroundColor = useMotionTemplate`rgba(9, 9, 11, ${bgOpacity})`;
  const backdropFilter = useMotionTemplate`blur(${blurValue}px)`;
  const padding = useMotionTemplate`${padValue}px 8px`;

  return (
    <motion.header
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ type: "spring", stiffness: 300, damping: 30 }}
      className="fixed top-8 inset-x-0 z-50 flex justify-center px-4 pointer-events-none"
    >
      <motion.nav
        layout
        style={{ backgroundColor, backdropFilter, padding }}
        className="pointer-events-auto w-full md:w-auto flex flex-col rounded-[2.5rem] border border-white/10 shadow-2xl shadow-black/40 overflow-hidden"
        transition={{ type: "spring", stiffness: 400, damping: 30 }}
      >
        <div className="flex items-center justify-between w-full">
          <Link
            href="/"
            className="px-4 py-2 md:mr-2 flex items-center gap-3 group overflow-hidden"
          >
            <Logo className="w-8 h-8 text-white group-hover:scale-105 transition-transform duration-300 shrink-0" />

            <div className="flex flex-col justify-center ml-1">
              <span className="font-bold text-white tracking-[0.15em] text-[15px] uppercase leading-none mb-1">
                Akssora
              </span>
              <span className="text-[10px] text-white/50 font-medium tracking-[0.25em] uppercase leading-none">
                UI Library
              </span>
            </div>
          </Link>

          <div className="hidden md:flex items-center gap-1 border-l border-white/10 pl-3">
            {nav_items.map((item, i) => {
              const isHovered = hoveredIndex === i;
              return (
                <motion.div
                  layout
                  key={item.title}
                  onMouseEnter={() => setHoveredIndex(i)}
                  onMouseLeave={() => setHoveredIndex(null)}
                  className={cn(
                    "relative flex items-center rounded-4xl cursor-pointer group/nav",
                    isHovered ? "text-white" : "text-white/60",
                  )}
                  // transition={{ type: "spring", stiffness: 400, damping: 30 }}
                >
                  {isHovered && (
                    <motion.div
                      layoutId="header-spotlight"
                      className="absolute inset-0 bg-white/10 rounded-full z-0 pointer-events-none"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      transition={{
                        type: "spring",
                        stiffness: 400,
                        damping: 30,
                      }}
                    />
                  )}
                  <Link
                    href={item.href}
                    className="relative z-10 flex items-center py-2 px-4 h-12 overflow-hidden"
                  >
                    <motion.span
                      layout
                      className="text-sm font-medium transition-colors whitespace-nowrap group-hover/nav:text-white text-white/60"
                    >
                      {item.title}
                    </motion.span>

                    <AnimatePresence mode="popLayout">
                      {isHovered && (
                        <motion.div
                          layout
                          initial={{ width: 0, opacity: 0, scale: 0.8 }}
                          animate={{ width: 180, opacity: 1, scale: 1 }}
                          exit={{ width: 0, opacity: 0, scale: 0.8 }}
                          transition={{
                            type: "spring",
                            stiffness: 400,
                            damping: 30,
                          }}
                          className="overflow-hidden relative flex items-center"
                        >
                          <div className="h-9 w-[170px] ml-3 rounded-full overflow-hidden relative shrink-0">
                            {/* eslint-disable-next-line @next/next/no-img-element */}
                            <img
                              src={item.image}
                              alt={item.title}
                              className="object-cover w-full h-full"
                            />
                            <div className="absolute inset-0 bg-black/40 flex items-center justify-between px-3 transition-colors hover:bg-black/20">
                              <span className="text-[11px] text-white font-semibold uppercase tracking-wider">
                                {item.description}
                              </span>
                              <ArrowUpRight className="w-3.5 h-3.5 text-white" />
                            </div>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </Link>
                </motion.div>
              );
            })}
          </div>

          <motion.button
            layout
            className="hidden md:block ml-3 mr-1 px-6 py-3 rounded-full bg-white text-black text-sm font-bold hover:scale-105 transition-transform duration-300 active:scale-95 whitespace-nowrap"
          >
            Get Template
          </motion.button>

          <motion.button
            layout
            onClick={() => setIsMobileOpen(!isMobileOpen)}
            className="md:hidden ml-4 mr-2 p-2 text-white hover:bg-white/10 rounded-full transition-colors"
          >
            {isMobileOpen ? (
              <X className="w-6 h-6" />
            ) : (
              <Menu className="w-6 h-6" />
            )}
          </motion.button>
        </div>

        <AnimatePresence>
          {isMobileOpen && (
            <motion.div
              layout
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ type: "spring", stiffness: 400, damping: 30 }}
              className="md:hidden flex flex-col px-4 pt-4 pb-2 gap-4 w-full"
            >
              {nav_items.map((item) => (
                <Link
                  key={item.title}
                  href={item.href}
                  onClick={() => setIsMobileOpen(false)}
                  className="flex items-center justify-between p-3 rounded-2xl hover:bg-white/10 transition-colors"
                >
                  <span className="text-white font-medium text-lg">
                    {item.title}
                  </span>
                  <ArrowUpRight className="w-5 h-5 text-white/50" />
                </Link>
              ))}
              <button className="w-full mt-2 py-4 rounded-xl bg-white text-black font-bold text-center active:scale-95 transition-transform">
                Get Template
              </button>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.nav>
    </motion.header>
  );
};

export default DynamicHeader;
