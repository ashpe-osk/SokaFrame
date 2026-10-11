import { useMutation, useQuery } from "@tanstack/react-query"
import { createFileRoute, Link } from "@tanstack/react-router"
import { ArrowLeft, Download } from "lucide-react"
import { useEffect, useState } from "react"

import {
  AnalysesService,
  type AnalysisStatus,
  type ArtifactKind,
} from "@/client"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"

export const Route = createFileRoute("/_layout/analyses/$analysisId")({
  component: AnalysisResultsPage,
  head: () => ({ meta: [{ title: "Analysis Results | SokaFrame" }] }),
})

const activeStatuses: AnalysisStatus[] = [
  "queued",
  "running",
  "cancel_requested",
]

const artifactDetails: Record<
  ArtifactKind,
  { label: string; fileName: string }
> = {
  tracked_video: { label: "Tracked video", fileName: "tracked.mp4" },
  browser_video: { label: "Browser video", fileName: "browser.mp4" },
  observations_csv: { label: "Observations", fileName: "observations.csv" },
  summary: { label: "Analysis summary (JSON)", fileName: "summary.json" },
  match_summary: {
    label: "Match summary (JSON)",
    fileName: "match_summary.json",
  },
  passes_csv: { label: "Passes", fileName: "passes.csv" },
  passing_network_csv: {
    label: "Passing network",
    fileName: "passing_network.csv",
  },
  team_statistics_csv: {
    label: "Team statistics",
    fileName: "team_statistics.csv",
  },
  player_statistics_csv: {
    label: "Player statistics",
    fileName: "player_statistics.csv",
  },
  possession_sequences_csv: {
    label: "Possession sequences",
    fileName: "possession_sequences.csv",
  },
  log: { label: "Processing log", fileName: "analysis.log" },
  other: { label: "Other artifact", fileName: "artifact.bin" },
}

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function formatElapsed(startedAt: string | null, finishedAt: string | null) {
  if (!startedAt) return "Not started"
  const end = finishedAt ? Date.parse(finishedAt) : Date.now()
  const seconds = Math.max(0, Math.floor((end - Date.parse(startedAt)) / 1000))
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`
}

function statusVariant(status: AnalysisStatus) {
  if (status === "failed" || status === "interrupted") return "destructive"
  if (status === "succeeded") return "secondary"
  return "outline"
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const link = document.createElement("a")
  link.href = url
  link.download = filename
  document.body.append(link)
  link.click()
  link.remove()
  window.setTimeout(() => URL.revokeObjectURL(url), 1000)
}

async function fetchArtifact(analysisId: string, kind: ArtifactKind) {
  const response = await AnalysesService.readAnalysisArtifact({
    path: { analysis_id: analysisId, kind },
    responseType: "blob",
  })
  if (!(response.data instanceof Blob)) {
    throw new Error("The server returned an invalid artifact response.")
  }
  return response.data
}

function AnalysisResultsPage() {
  const { analysisId } = Route.useParams()
  const [videoUrl, setVideoUrl] = useState("")
  const [videoPlaybackError, setVideoPlaybackError] = useState(false)
  const analysisQuery = useQuery({
    queryKey: ["analysis", analysisId],
    queryFn: async () =>
      (
        await AnalysesService.readAnalysis({
          path: { analysis_id: analysisId },
        })
      ).data,
    refetchInterval: (query) =>
      query.state.data && activeStatuses.includes(query.state.data.status)
        ? 2500
        : false,
    refetchIntervalInBackground: true,
  })
  const analysis = analysisQuery.data
  const browserVideo = analysis?.artifacts?.find(
    (artifact) => artifact.kind === "browser_video",
  )
  const videoQuery = useQuery({
    queryKey: ["analysis-artifact", analysisId, "browser_video"],
    queryFn: () => fetchArtifact(analysisId, "browser_video"),
    enabled: analysis?.status === "succeeded" && Boolean(browserVideo),
    retry: false,
  })
  const downloadMutation = useMutation({
    mutationFn: (kind: ArtifactKind) => fetchArtifact(analysisId, kind),
    onSuccess: (blob, kind) =>
      downloadBlob(blob, artifactDetails[kind].fileName),
  })
  const serializedSummary = analysis?.summary
    ? JSON.stringify(analysis.summary, null, 2)
    : null

  useEffect(() => {
    if (!videoQuery.data) {
      setVideoUrl("")
      setVideoPlaybackError(false)
      return
    }
    const url = URL.createObjectURL(videoQuery.data)
    setVideoUrl(url)
    setVideoPlaybackError(false)
    return () => URL.revokeObjectURL(url)
  }, [videoQuery.data])

  if (analysisQuery.isPending) return <p role="status">Loading analysis…</p>
  if (analysisQuery.isError) {
    return (
      <Alert variant="destructive">
        <AlertTitle>Analysis unavailable</AlertTitle>
        <AlertDescription>{analysisQuery.error.message}</AlertDescription>
      </Alert>
    )
  }
  if (!analysis) {
    return (
      <Alert variant="destructive">
        <AlertTitle>Analysis unavailable</AlertTitle>
        <AlertDescription>This analysis could not be found.</AlertDescription>
      </Alert>
    )
  }

  const isComplete = analysis.status === "succeeded"
  const percent =
    analysis.total_frames && analysis.total_frames > 0
      ? Math.min(
          100,
          Math.floor((analysis.frames_processed / analysis.total_frames) * 100),
        )
      : null

  return (
    <div className="flex flex-col gap-6">
      <Link
        className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground"
        params={{ projectId: analysis.project_id }}
        to="/projects/$projectId"
      >
        <ArrowLeft className="size-4" />
        Back to project
      </Link>

      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">
            Analysis results
          </h1>
          <p className="text-muted-foreground">
            {analysis.video_original_filename}
          </p>
        </div>
        <Badge variant={statusVariant(analysis.status)}>
          {analysis.status}
        </Badge>
      </header>

      {activeStatuses.includes(analysis.status) ? (
        <Card>
          <CardHeader>
            <CardTitle>Analysis in progress</CardTitle>
            <CardDescription>
              Results will appear here after processing completes.
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-3">
            {analysis.total_frames !== null && (
              <>
                <div className="flex justify-between text-sm">
                  <span>
                    {analysis.frames_processed} / {analysis.total_frames} frames
                  </span>
                  <span>{percent === null ? "—" : `${percent}%`}</span>
                </div>
                <progress
                  aria-label="Analysis progress"
                  className="h-2 w-full accent-primary"
                  max={analysis.total_frames || 1}
                  value={analysis.frames_processed}
                />
              </>
            )}
            <p className="text-sm text-muted-foreground">
              Elapsed: {formatElapsed(analysis.started_at, null)}
            </p>
          </CardContent>
        </Card>
      ) : !isComplete ? (
        <Alert
          variant={analysis.status === "failed" ? "destructive" : "default"}
        >
          <AlertTitle>Results are unavailable</AlertTitle>
          <AlertDescription>
            This analysis ended with status “{analysis.status}”. A results
            viewer is only available after a successful analysis.
            {analysis.error_code ? ` Error code: ${analysis.error_code}.` : ""}
          </AlertDescription>
        </Alert>
      ) : (
        <>
          <Card>
            <CardHeader>
              <CardTitle>Processed video</CardTitle>
              <CardDescription>
                {browserVideo
                  ? `${browserVideo.media_type} · ${formatBytes(browserVideo.file_size_bytes)}`
                  : "The processed video artifact is not available."}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {!browserVideo ? (
                <p role="status" className="text-sm text-muted-foreground">
                  Processed video is unavailable for this analysis.
                </p>
              ) : videoQuery.isPending ? (
                <p role="status">Loading processed video…</p>
              ) : videoQuery.isError ? (
                <Alert variant="destructive">
                  <AlertTitle>Processed video could not be loaded</AlertTitle>
                  <AlertDescription>
                    {videoQuery.error.message}
                  </AlertDescription>
                </Alert>
              ) : videoPlaybackError ? (
                <div className="grid gap-2">
                  <p className="text-sm text-destructive" role="alert">
                    This browser could not play the processed video. Download
                    the artifact to open it in a compatible player.
                  </p>
                  {videoQuery.data && (
                    <Button
                      className="w-fit"
                      onClick={() =>
                        downloadBlob(videoQuery.data, "browser.mp4")
                      }
                      type="button"
                      variant="outline"
                    >
                      <Download />
                      Download Processed Video
                    </Button>
                  )}
                </div>
              ) : !videoUrl ? (
                <p role="status">Preparing processed video…</p>
              ) : (
                // biome-ignore lint/a11y/useMediaCaption: Phase 6 does not produce a caption artifact.
                <video
                  aria-label="Processed analysis video"
                  className="max-h-[70vh] w-full rounded-md bg-black"
                  controls
                  onError={() => setVideoPlaybackError(true)}
                  playsInline
                  preload="metadata"
                  src={videoUrl}
                >
                  Your browser cannot play this video format.
                </video>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Processing information</CardTitle>
            </CardHeader>
            <CardContent className="grid gap-2 text-sm sm:grid-cols-2">
              <p>Frames processed: {analysis.frames_processed}</p>
              <p>Total frames: {analysis.total_frames ?? "Unavailable"}</p>
              <p>
                Elapsed:{" "}
                {formatElapsed(analysis.started_at, analysis.finished_at)}
              </p>
              <p>Source size: {formatBytes(analysis.video_size_bytes)}</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Downloads</CardTitle>
              <CardDescription>
                Download files produced by this analysis.
              </CardDescription>
            </CardHeader>
            <CardContent className="grid gap-3">
              {analysis.artifacts?.length ? (
                <>
                  <ul className="grid gap-3">
                    {analysis.artifacts.map((artifact) => {
                      const details = artifactDetails[artifact.kind]
                      const isDownloading =
                        downloadMutation.isPending &&
                        downloadMutation.variables === artifact.kind

                      return (
                        <li
                          className="flex flex-wrap items-center justify-between gap-3 rounded-md border p-3"
                          key={artifact.id}
                        >
                          <div className="grid gap-1">
                            <span className="font-medium">{details.label}</span>
                            <span className="text-sm text-muted-foreground">
                              {artifact.media_type} ·{" "}
                              {formatBytes(artifact.file_size_bytes)}
                            </span>
                          </div>
                          <Button
                            disabled={downloadMutation.isPending}
                            onClick={() =>
                              downloadMutation.mutate(artifact.kind)
                            }
                            type="button"
                            variant="outline"
                          >
                            <Download />
                            {isDownloading ? "Preparing download…" : "Download"}
                          </Button>
                        </li>
                      )
                    })}
                  </ul>
                  {downloadMutation.isError && (
                    <p className="text-sm text-destructive" role="alert">
                      Download failed: {downloadMutation.error.message}
                    </p>
                  )}
                </>
              ) : (
                <p role="status" className="text-sm text-muted-foreground">
                  No downloadable files are available for this analysis.
                </p>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Analysis summary</CardTitle>
            </CardHeader>
            <CardContent>
              {serializedSummary ? (
                <details
                  className="group rounded-md border px-4 py-3"
                  open={serializedSummary.length <= 1200}
                >
                  <summary className="cursor-pointer font-medium">
                    {serializedSummary.length > 1200
                      ? "Show detailed analysis summary"
                      : "Analysis summary details"}
                  </summary>
                  <pre className="mt-3 max-h-[32rem] overflow-auto rounded-md bg-muted p-4 text-xs">
                    {serializedSummary}
                  </pre>
                </details>
              ) : (
                <p role="status" className="text-sm text-muted-foreground">
                  Summary is unavailable for this analysis.
                </p>
              )}
            </CardContent>
          </Card>
        </>
      )}
    </div>
  )
}
