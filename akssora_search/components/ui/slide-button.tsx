"use client";

import { useRef } from "react";
import { cn } from "@/lib/utils";
import { motion, useMotionValue, useSpring } from "motion/react";

export function SlideButton({
  className,
  children,
  onClick,
  disabled,
}: {
  className?: string;
  children: React.ReactNode;
  onClick?: React.MouseEventHandler<HTMLButtonElement>;
  disabled?: boolean;
}) {
  const buttonRef = useRef<HTMLButtonElement>(null);

  // Magnetic physics
  const x = useMotionValue(0);
  const y = useMotionValue(0);

  const springConfig = { stiffness: 150, damping: 15, mass: 0.1 };
  const springX = useSpring(x, springConfig);
  const springY = useSpring(y, springConfig);

  const handleMouseMove = (e: React.MouseEvent<HTMLButtonElement>) => {
    if (!buttonRef.current) return;
    const rect = buttonRef.current.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    // Calculate distance from center, scale it down for a subtle magnetic pull (e.g., 20% of distance)
    const distanceX = (e.clientX - centerX) * 0.2;
    const distanceY = (e.clientY - centerY) * 0.2;
    x.set(distanceX);
    y.set(distanceY);
  };

  const handleMouseLeave = () => {
    // Elastic snap back to origin
    x.set(0);
    y.set(0);
  };

  return (
    <motion.button
      ref={buttonRef}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      style={{ x: springX, y: springY }}
      className={cn(
        "group w-fit px-8 py-4 flex items-center justify-center cursor-pointer relative overflow-hidden rounded-full border border-black/10 bg-white text-black font-semibold text-sm uppercase tracking-widest",
        "transition-shadow hover:shadow-[0_10px_40px_rgba(0,0,0,0.1)]",
        className,
      )}
      onClick={onClick}
      disabled={disabled}
    >
      {/* Expanding Liquid Background Fill */}
      <span className="absolute inset-0 bg-black translate-y-[105%] group-hover:translate-y-0 transition-transform duration-500 ease-[cubic-bezier(0.19,1,0.22,1)] rounded-full" />

      {/* Inner Label Container */}
      <motion.span className="relative z-10 transition-colors duration-300 ease-out group-hover:text-white">
        {children}
      </motion.span>
    </motion.button>
  );
}
