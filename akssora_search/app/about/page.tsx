"use client";

import React, { useRef } from "react";
import { motion, useInView } from "motion/react";
import {
  Search,
  Zap,
  Layers,
  Clock,
  Tag,
  BarChart3,
  Mic,
  GraduationCap,
  ArrowRight,
} from "lucide-react";

/* ─────────────────────────────────────────────
   Fade-in wrapper - triggers when element enters viewport
───────────────────────────────────────────── */
const FadeIn = ({
  children,
  delay = 0,
  className,
}: {
  children: React.ReactNode;
  delay?: number;
  className?: string;
}) => {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-60px" });
  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 28 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.55, delay, ease: [0.22, 1, 0.36, 1] }}
      className={className}
    >
      {children}
    </motion.div>
  );
};

/* ─────────────────────────────────────────────
   Feature card
───────────────────────────────────────────── */
const FeatureCard = ({
  icon: Icon,
  title,
  description,
  delay,
}: {
  icon: React.ElementType;
  title: string;
  description: string;
  delay?: number;
}) => (
  <FadeIn delay={delay} className="h-full">
    <div className="h-full rounded-2xl border border-border bg-muted/30 p-6 hover:bg-muted/50 hover:border-ring/30 transition-all duration-300 group">
      <div className="w-10 h-10 rounded-xl bg-primary/10 border border-primary/20 flex items-center justify-center mb-4 group-hover:bg-primary/20 transition-colors duration-300">
        <Icon className="w-5 h-5 text-primary" />
      </div>
      <h3 className="text-sm font-semibold text-foreground mb-2">{title}</h3>
      <p className="text-sm text-muted-foreground leading-relaxed">
        {description}
      </p>
    </div>
  </FadeIn>
);

/* ─────────────────────────────────────────────
   Audience card
───────────────────────────────────────────── */
const AudienceCard = ({
  icon: Icon,
  title,
  description,
  delay,
}: {
  icon: React.ElementType;
  title: string;
  description: string;
  delay?: number;
}) => (
  <FadeIn delay={delay}>
    <div className="rounded-3xl border border-border bg-muted/20 p-8 hover:border-ring/40 transition-all duration-300">
      <div className="flex items-center gap-3 mb-4">
        <div className="w-11 h-11 rounded-2xl bg-primary/10 border border-primary/20 flex items-center justify-center">
          <Icon className="w-5 h-5 text-primary" />
        </div>
        <h3 className="text-base font-semibold text-foreground">{title}</h3>
      </div>
      <p className="text-sm text-muted-foreground leading-relaxed">
        {description}
      </p>
    </div>
  </FadeIn>
);

/* ─────────────────────────────────────────────
   Step row
───────────────────────────────────────────── */
const Step = ({
  number,
  title,
  description,
  delay,
}: {
  number: string;
  title: string;
  description: string;
  delay?: number;
}) => (
  <FadeIn delay={delay} className="flex gap-5">
    <div className="flex-shrink-0 w-10 h-10 rounded-full border border-primary/30 bg-primary/10 flex items-center justify-center">
      <span className="text-xs font-mono font-bold text-primary">{number}</span>
    </div>
    <div className="pt-1.5">
      <h4 className="text-sm font-semibold text-foreground mb-1">{title}</h4>
      <p className="text-sm text-muted-foreground leading-relaxed">
        {description}
      </p>
    </div>
  </FadeIn>
);

/* ─────────────────────────────────────────────
   Divider
───────────────────────────────────────────── */
const Divider = () => (
  <div className="w-full h-px bg-gradient-to-r from-transparent via-border to-transparent my-20" />
);

/* ─────────────────────────────────────────────
   Main page
───────────────────────────────────────────── */
const AboutPage = () => {
  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* ── Hero ── */}
      <section className="relative max-w-5xl mx-auto px-4 sm:px-6 pt-32 pb-24 text-center overflow-hidden">
        {/* Glow behind heading */}
        <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
          <div className="w-[600px] h-[300px] rounded-full bg-primary/5 blur-3xl" />
        </div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-primary/20 bg-primary/10 text-xs font-mono text-primary mb-6"
        >
          <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
          Powered by Amazon Bedrock
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1, ease: [0.22, 1, 0.36, 1] }}
          className="text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight text-foreground mb-6 leading-[1.1]"
        >
          Find any moment.
          <br />
          <span className="text-primary">Across every video.</span>
          <br />
          Instantly.
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2, ease: [0.22, 1, 0.36, 1] }}
          className="text-base sm:text-lg text-muted-foreground max-w-2xl mx-auto leading-relaxed"
        >
          Akssora Search is an AI-powered media search engine purpose-built for
          podcasters and educators - making every frame, every word, and every
          moment in your content library discoverable.
        </motion.p>
      </section>

      <div className="max-w-5xl mx-auto px-4 sm:px-6">
        <Divider />

        {/* ── Problem ── */}
        <section className="mb-20">
          <FadeIn>
            <p className="text-xs font-mono text-primary tracking-widest uppercase mb-4">
              The Problem
            </p>
            <h2 className="text-2xl sm:text-3xl font-bold text-foreground mb-6 max-w-2xl">
              Your content is valuable. But most of it is invisible.
            </h2>
          </FadeIn>
          <FadeIn delay={0.1}>
            <p className="text-muted-foreground leading-relaxed max-w-3xl mb-4">
              A podcaster with 300 episodes cannot recall which conversation
              touched on a specific topic. An edtech platform with thousands of
              lecture recordings cannot surface the exact moment a concept was
              explained. Traditional search fails here - it relies on tags,
              titles, and metadata that no one has time to maintain.
            </p>
            <p className="text-muted-foreground leading-relaxed max-w-3xl">
              Akssora Search eliminates that friction entirely. Type what you
              remember seeing or hearing, and we find it - no tags, no manual
              labelling, no guesswork.
            </p>
          </FadeIn>
        </section>

        <Divider />

        {/* ── How it works ── */}
        <section className="mb-20">
          <FadeIn>
            <p className="text-xs font-mono text-primary tracking-widest uppercase mb-4">
              How It Works
            </p>
            <h2 className="text-2xl sm:text-3xl font-bold text-foreground mb-12 max-w-2xl">
              Two indexes. One search. Precise results.
            </h2>
          </FadeIn>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-10 items-start">
            <div className="flex flex-col gap-8">
              <Step
                number="01"
                title="Upload your media"
                description="Every image and video you upload is automatically processed - no manual configuration required."
                delay={0.05}
              />
              <Step
                number="02"
                title="AI builds two indexes"
                description="Visual embeddings capture what is seen. Text embeddings capture what is said or described. Both are stored independently."
                delay={0.1}
              />
              <Step
                number="03"
                title="You type naturally"
                description="Describe the scene, the topic, or the moment. Akssora expands your query into multiple visual and textual search angles automatically."
                delay={0.15}
              />
              <Step
                number="04"
                title="Results ranked by relevance"
                description="Visual and text results are merged using Reciprocal Rank Fusion, then re-ranked by an AI relevance judge - surfacing the best match first."
                delay={0.2}
              />
            </div>

            {/* Visual diagram */}
            <FadeIn delay={0.15}>
              <div className="rounded-3xl border border-border bg-muted/20 p-6 space-y-4">
                {[
                  {
                    label: "Query",
                    value: "man coming out of metro station",
                    icon: Search,
                  },
                  {
                    label: "Visual Index",
                    value: "Scene · Objects · Colors · Actions",
                    icon: Layers,
                  },
                  {
                    label: "Text Index",
                    value: "Transcripts · Descriptions",
                    icon: Mic,
                  },
                  {
                    label: "RRF Merge + Rerank",
                    value: "Best moments surfaced first",
                    icon: BarChart3,
                  },
                ].map(({ label, value, icon: Icon }, i) => (
                  <div key={i}>
                    <div className="flex items-center gap-3 py-3 px-4 rounded-xl bg-background/60 border border-border">
                      <Icon className="w-4 h-4 text-primary flex-shrink-0" />
                      <div>
                        <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-wider">
                          {label}
                        </p>
                        <p className="text-xs text-foreground font-medium">
                          {value}
                        </p>
                      </div>
                    </div>
                    {i < 3 && (
                      <div className="flex justify-center py-1">
                        <ArrowRight className="w-3 h-3 text-muted-foreground rotate-90" />
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </FadeIn>
          </div>
        </section>

        <Divider />

        {/* ── Features ── */}
        <section className="mb-20">
          <FadeIn>
            <p className="text-xs font-mono text-primary tracking-widest uppercase mb-4">
              Capabilities
            </p>
            <h2 className="text-2xl sm:text-3xl font-bold text-foreground mb-10 max-w-2xl">
              Built differently from the ground up.
            </h2>
          </FadeIn>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            <FeatureCard
              icon={Search}
              title="Visual-first search"
              description="Search by describing a scene - colors, objects, actions, settings. No filename required."
              delay={0.05}
            />
            <FeatureCard
              icon={Clock}
              title="Moment-level precision"
              description="We don't just find the right video - we find the right timestamp inside it. Every segment is independently indexed."
              delay={0.1}
            />
            <FeatureCard
              icon={Layers}
              title="Multimodal by design"
              description="Images and videos live in the same index. One query surfaces everything relevant regardless of format."
              delay={0.15}
            />
            <FeatureCard
              icon={Tag}
              title="Zero manual tagging"
              description="Upload and your content is immediately searchable. The AI handles analysis, description, and indexing automatically."
              delay={0.2}
            />
            <FeatureCard
              icon={Zap}
              title="Parallel search architecture"
              description="Visual and text indexes are queried simultaneously. Results are merged and re-ranked in a single fast pipeline."
              delay={0.25}
            />
            <FeatureCard
              icon={BarChart3}
              title="Scales with your library"
              description="Purpose-built for large, growing content catalogs. Performance does not degrade as your archive expands."
              delay={0.3}
            />
          </div>
        </section>

        <Divider />

        {/* ── Audience ── */}
        <section className="mb-20">
          <FadeIn>
            <p className="text-xs font-mono text-primary tracking-widest uppercase mb-4">
              Who It's For
            </p>
            <h2 className="text-2xl sm:text-3xl font-bold text-foreground mb-10 max-w-2xl">
              Built for creators who think in moments.
            </h2>
          </FadeIn>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            <AudienceCard
              icon={Mic}
              title="Podcasters"
              description="Your back catalog is an untapped asset. Akssora Search lets you, your editors, and your audience navigate hundreds of hours of conversation by simply describing what they're looking for. Find the moment a guest said something memorable. Clip it. Ship it."
              delay={0.05}
            />
            <AudienceCard
              icon={GraduationCap}
              title="EdTech Platforms"
              description="Lecture libraries grow fast and become impossible to navigate. Akssora Search gives students and instructors the ability to search by concept, scene, or spoken phrase - surfacing the exact timestamp where a topic was covered, not just the video it might be in."
              delay={0.1}
            />
          </div>
        </section>

        <Divider />

        {/* ── Mission ── */}
        <section className="mb-20 text-center">
          <FadeIn>
            <p className="text-xs font-mono text-primary tracking-widest uppercase mb-6">
              Our Mission
            </p>
            <blockquote className="text-2xl sm:text-3xl font-semibold text-foreground max-w-3xl mx-auto leading-snug mb-6">
              "The value locked inside video content is massively underutilised
              - not because the content isn't good, but because it's invisible
              to search."
            </blockquote>
            <p className="text-muted-foreground max-w-xl mx-auto leading-relaxed">
              Akssora Search makes every frame, every word, and every moment in
              your library discoverable. Your content works hard to be created.
              It should work just as hard to be found.
            </p>
          </FadeIn>
        </section>

        <Divider />

        {/* ── Stack ── */}
        <section className="mb-24">
          <FadeIn>
            <p className="text-xs font-mono text-muted-foreground tracking-widest uppercase mb-6 text-center">
              Built on enterprise-grade infrastructure
            </p>
            <div className="flex flex-wrap justify-center gap-3">
              {[
                "Amazon Bedrock",
                "Amazon Nova",
                "Titan Text Embeddings",
                "S3 Vectors",
                "FastAPI",
                "Next.js",
              ].map((tech) => (
                <span
                  key={tech}
                  className="px-4 py-2 rounded-full border border-border bg-muted/30 text-xs font-mono text-muted-foreground hover:text-foreground hover:border-ring/40 transition-colors duration-200"
                >
                  {tech}
                </span>
              ))}
            </div>
          </FadeIn>
        </section>
      </div>
    </div>
  );
};

export default AboutPage;
