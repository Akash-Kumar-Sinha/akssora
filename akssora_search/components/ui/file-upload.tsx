"use client";

import React, { useRef, useState, useEffect } from "react";
import { motion, AnimatePresence } from "motion/react";
import { useDropzone, DropzoneOptions } from "react-dropzone";
import { UploadCloud, File as FileIcon, X, Plus } from "lucide-react";
import { cn } from "@/lib/utils";

interface FileUploadProps extends DropzoneOptions {
  className?: string;
  value?: File[];
  onChange?: (files: File[]) => void;
  maxFiles?: number;
}

function FileThumbnail({ file }: { file: File }) {
  const [url, setUrl] = useState<string | null>(null);

  useEffect(() => {
    if (file.type.startsWith("image/") || file.type.startsWith("video/")) {
      const objectUrl = URL.createObjectURL(file);
      setUrl(objectUrl);
      return () => URL.revokeObjectURL(objectUrl);
    }
  }, [file]);

  if (!url) return <FileIcon className="w-3.5 h-3.5 text-zinc-500" />;
  if (file.type.startsWith("image/"))
    return (
      <img src={url} alt={file.name} className="w-full h-full object-cover" />
    );
  if (file.type.startsWith("video/"))
    return (
      <video
        src={url}
        className="w-full h-full object-cover"
        muted
        playsInline
      />
    );
  return <FileIcon className="w-3.5 h-3.5 text-zinc-500" />;
}

export function FileUpload({
  className,
  value,
  onChange,
  maxFiles = Infinity,
  ...dropzoneProps
}: FileUploadProps) {
  const [files, setFiles] = useState<File[]>(value || []);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && selectedFile) {
        setSelectedFile(null);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [selectedFile]);

  useEffect(() => {
    if (errorMessage) {
      const timer = setTimeout(() => {
        setErrorMessage(null);
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [errorMessage]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    ...dropzoneProps,
    maxFiles,
    onDragEnter: (event) => {
      const dt = (event as React.DragEvent<HTMLElement>).dataTransfer;
      if (dt?.items) {
        let count = 0;
        for (let i = 0; i < dt.items.length; i++) {
          if (dt.items[i].kind === "file") count++;
        }
        if (files.length + count > maxFiles) {
          setErrorMessage(
            `Cannot upload more than ${maxFiles} file${maxFiles === 1 ? "" : "s"}.`,
          );
        }
      }
    },
    onDrop: (acceptedFiles, fileRejections, event) => {
      const total = files.length + acceptedFiles.length;
      if (total > maxFiles) {
        setErrorMessage(
          `Cannot upload more than ${maxFiles} file${maxFiles === 1 ? "" : "s"}.`,
        );
      }

      const remainingSlots = maxFiles - files.length;
      const filesToAdd = acceptedFiles.slice(0, remainingSlots);

      if (filesToAdd.length > 0) {
        const newFiles = [...files, ...filesToAdd];
        setFiles(newFiles);
        onChange?.(newFiles);
      }

      if (dropzoneProps.onDrop) {
        dropzoneProps.onDrop(acceptedFiles, fileRejections, event);
      }
    },
  });

  const removeFile = (e: React.MouseEvent, index: number) => {
    e.stopPropagation();
    const fileToRemove = files[index];
    const newFiles = files.filter((_, i) => i !== index);
    setFiles(newFiles);
    onChange?.(newFiles);
    if (selectedFile === fileToRemove) {
      setSelectedFile(null);
    }
  };

  const handleFileClick = (e: React.MouseEvent, file: File) => {
    e.stopPropagation();
    if (file.type.startsWith("image/") || file.type.startsWith("video/")) {
      setSelectedFile(file);
      setErrorMessage(null);
    } else {
      setErrorMessage(`Cannot preview ${file.type || "this"} file format.`);
      setSelectedFile(null);
    }
  };

  const hasReachedMax = files.length >= maxFiles;

  return (
    <div className={cn("w-full relative", className)}>
      <div
        ref={containerRef}
        {...getRootProps()}
        className={cn(
          "relative flex flex-wrap items-center gap-2 p-2 min-h-20 rounded-4xl border-2 transition-all duration-200 ease-out overflow-hidden",
          "bg-white shadow-[0_8px_30px_rgb(0,0,0,0.04)]",
          isDragActive
            ? "border-black"
            : hasReachedMax
              ? "border-black/5 cursor-default bg-zinc-50"
              : "border-black/10 hover:border-black/20 cursor-pointer hover:bg-zinc-50 hover:shadow-[0_8px_30px_rgb(0,0,0,0.08)]",
        )}
      >
        <input {...getInputProps()} disabled={hasReachedMax} />

        <AnimatePresence mode="popLayout">
          {files.map((file, idx) => (
            <motion.div
              layout
              initial={{ opacity: 0, scale: 0.9, y: 10 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.9, y: 10 }}
              transition={{
                type: "spring",
                stiffness: 500,
                damping: 30,
                delay: (idx % 15) * 0.04,
              }}
              key={`${file.name}-${idx}`}
              onClick={(e) => handleFileClick(e, file)}
              className={cn(
                "group relative flex items-center gap-2 bg-zinc-100 hover:bg-zinc-200 transition-colors rounded-full px-3 py-1.5 pr-2 max-w-[200px] cursor-pointer",
                isDragActive && "pointer-events-none",
              )}
            >
              <div className="flex items-center justify-center w-6 h-6 bg-white rounded-full shrink-0 shadow-sm border border-black/5 overflow-hidden">
                <FileThumbnail file={file} />
              </div>

              <span className="text-sm font-medium text-zinc-700 truncate">
                {file.name}
              </span>

              <button
                type="button"
                onClick={(e) => removeFile(e, idx)}
                className="ml-1 p-1 rounded-full text-zinc-400 hover:text-red-500 hover:bg-black/5 dark:hover:bg-white/10 transition-colors focus:outline-none"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </motion.div>
          ))}
        </AnimatePresence>

        <motion.div
          layout
          className={cn(
            "flex-1 min-w-[200px] flex items-center gap-3 px-3 py-2",
            isDragActive && "pointer-events-none",
          )}
        >
          <AnimatePresence mode="popLayout">
            {isDragActive ? (
              <motion.div
                key="drag-prompt"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ type: "spring", stiffness: 500, damping: 30 }}
                className="flex items-center gap-3"
              >
                <motion.div
                  animate={{ y: [0, -5, 0] }}
                  transition={{
                    duration: 1.6,
                    repeat: Infinity,
                    ease: "easeInOut",
                  }}
                  className="flex items-center justify-center w-8 h-8 rounded-full bg-black text-white shrink-0"
                >
                  <UploadCloud className="w-4 h-4" />
                </motion.div>
                <div>
                  <p className="text-sm font-semibold text-black">
                    Drop files to upload
                  </p>
                  <p className="text-xs text-black/40 mt-0.5">
                    Release anywhere to add your files
                  </p>
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="normal-prompt"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ type: "spring", stiffness: 500, damping: 30 }}
                className="flex items-center gap-3"
              >
                {!hasReachedMax && (
                  <div
                    className={cn(
                      "flex items-center justify-center w-8 h-8 rounded-full transition-transform duration-300",
                      files.length > 0
                        ? "bg-black/5 text-zinc-600"
                        : "bg-black text-white",
                    )}
                  >
                    {files.length > 0 ? (
                      <Plus className={cn("w-4 h-4 text-zinc-600")} />
                    ) : (
                      <UploadCloud className="w-4 h-4" />
                    )}
                  </div>
                )}

                <div className="flex flex-col">
                  <span className="text-sm font-semibold text-zinc-700">
                    {hasReachedMax
                      ? "Upload limit reached"
                      : files.length > 0
                        ? "Add more files"
                        : "Upload files"}
                  </span>
                  {!hasReachedMax && files.length === 0 && (
                    <span className="text-xs text-zinc-400">
                      Drag and drop or click to browse
                    </span>
                  )}
                  {hasReachedMax && (
                    <span className="text-xs text-zinc-400">
                      {files.length} of {maxFiles} files uploaded
                    </span>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </div>

      <AnimatePresence>
        {errorMessage && (
          <motion.div
            initial={{ opacity: 0, y: 10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.95 }}
            className="absolute left-1/2 -translate-x-1/2 -bottom-12 z-50 pointer-events-none"
          >
            <div className="bg-zinc-900 text-zinc-50 dark:bg-zinc-100 dark:text-zinc-900 px-4 py-2.5 rounded-full text-xs font-semibold shadow-xl flex items-center gap-2 border border-black/10 dark:border-white/10">
              <FileIcon className="w-4 h-4" />
              {errorMessage}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {selectedFile && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="absolute inset-0 bg-black/50 backdrop-blur-xl"
              onClick={() => setSelectedFile(null)}
            />

            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              transition={{ type: "spring", stiffness: 400, damping: 32 }}
              className="relative z-10 w-full max-w-2xl bg-white rounded-2xl overflow-hidden shadow-[0_24px_60px_rgba(0,0,0,0.18)] flex flex-col"
            >
              <div className="flex items-center justify-between gap-3 px-4 py-3 bg-white border-b border-zinc-100">
                <div className="flex items-center gap-2.5 min-w-0">
                  <div className="flex items-center justify-center w-7 h-7 rounded-lg bg-zinc-100 shrink-0">
                    <FileIcon className="w-3.5 h-3.5 text-zinc-500" />
                  </div>
                  <span className="text-sm font-medium text-zinc-700 truncate">
                    {selectedFile.name}
                  </span>
                </div>
                <button
                  onClick={() => setSelectedFile(null)}
                  className="w-7 h-7 flex items-center justify-center rounded-full text-zinc-400 hover:text-zinc-700 hover:bg-zinc-100 transition-colors shrink-0"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
              {/* Media */}
              <div className="bg-zinc-50 flex items-center justify-center max-h-[70vh] overflow-hidden">
                {selectedFile.type.startsWith("image/") ? (
                  <img
                    src={URL.createObjectURL(selectedFile)}
                    alt={selectedFile.name}
                    className="max-w-full max-h-[70vh] object-contain"
                  />
                ) : selectedFile.type.startsWith("video/") ? (
                  <video
                    src={URL.createObjectURL(selectedFile)}
                    controls
                    autoPlay
                    className="max-w-full max-h-[70vh] w-full"
                  />
                ) : null}
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}
