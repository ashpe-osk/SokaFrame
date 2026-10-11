import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { createFileRoute, Link } from "@tanstack/react-router"
import { ArrowLeft, Ban, ExternalLink, Upload } from "lucide-react"
import { useState } from "react"

import {
  AnalysesService,
  type AnalysisPublic,
  type AnalysisStatus,
  ProjectsService,
} from "@/client"
import { client } from "@/client/client.gen"
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
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

export const Route = createFileRoute("/_layout/projects/$projectId")({
  component: ProjectDetailsPage,
  head: () => ({ meta: [{ title: "Project | SokaFrame" }] }),
})

const activeStatuses: AnalysisStatus[] = [
  "queued",
  "running",
  "cancel_requested",
]

function formatElapsed(startedAt: string | null, finishedAt: string | null) {
  if (!startedAt) return "—"
  const end = finishedAt ? Date.parse(finishedAt) : Date.now()
  const seconds = Math.max(0, Math.floor((end - Date.parse(startedAt)) / 1000))
  const minutes = Math.floor(seconds / 60)
  return `${minutes}:${String(seconds % 60).padStart(2, "0")}`
}

function statusVariant(status: AnalysisStatus) {
  if (status === "failed" || status === "interrupted") return "destructive"
  if (status === "succeeded") return "secondary"
  return "outline"
}

function ProjectDetailsPage() {
  const { projectId } = Route.useParams()
  const queryClient = useQueryClient()
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [uploadProgress, setUploadProgress] = useState<number | null>(null)
  const [validationError, setValidationError] = useState("")
  const projectQuery = useQuery({
    queryKey: ["project", projectId],
    queryFn: async () =>
      (
        await ProjectsService.readProject({
          path: { project_id: projectId },
        })
      ).data,
  })
  const analysesQuery = useQuery({
    queryKey: ["analyses", projectId],
    queryFn: async () =>
      (
        await AnalysesService.readAnalyses({
          query: { skip: 0, limit: 100 },
        })
      ).data.data.filter((analysis) => analysis.project_id === projectId),
    refetchInterval: (query) =>
      query.state.data?.some((analysis) =>
        activeStatuses.includes(analysis.status),
      )
        ? 2500
        : false,
  })

  const uploadMutation = useMutation({
    mutationFn: (file: File) => {
      setUploadProgress(0)
      return AnalysesService.createAnalysis({
        body: { project_id: projectId, video_file: file },
        onUploadProgress: (event) => {
          if (event.total) {
            setUploadProgress(
              Math.min(100, Math.round((event.loaded / event.total) * 100)),
            )
          }
        },
      })
    },
    onSuccess: async () => {
      setSelectedFile(null)
      setUploadProgress(null)
      const fileInput = document.getElementById(
        "source-video",
      ) as HTMLInputElement | null
      if (fileInput) fileInput.value = ""
      await queryClient.invalidateQueries({ queryKey: ["analyses", projectId] })
    },
  })

  const cancelMutation = useMutation({
    mutationFn: (analysisId: string) =>
      client.post({
        url: `/api/v1/analyses/${analysisId}/cancel`,
        security: [{ scheme: "bearer", type: "http" }],
        responseType: "json",
      }),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["analyses", projectId] }),
  })

  const upload = () => {
    setValidationError("")
    if (!selectedFile) {
      setValidationError("Choose a video file to upload.")
      return
    }
    if (!/\.(mp4|mov|avi)$/i.test(selectedFile.name)) {
      setValidationError("Choose an MP4, MOV, or AVI video.")
      return
    }
    if (selectedFile.size === 0) {
      setValidationError("The selected video is empty.")
      return
    }
    setUploadProgress(0)
    uploadMutation.mutate(selectedFile)
  }

  if (projectQuery.isPending) return <p role="status">Loading project…</p>
  if (projectQuery.isError) {
    return (
      <Alert variant="destructive">
        <AlertTitle>Project unavailable</AlertTitle>
        <AlertDescription>{projectQuery.error.message}</AlertDescription>
      </Alert>
    )
  }

  return (
    <div className="flex flex-col gap-6">
      <Link
        className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground"
        to="/projects"
      >
        <ArrowLeft className="size-4" />
        All projects
      </Link>
      <header>
        <h1 className="text-2xl font-bold tracking-tight">
          {projectQuery.data.name}
        </h1>
        <p className="text-muted-foreground">
          {projectQuery.data.description ||
            "Manage source videos and analyses."}
        </p>
      </header>

      <Card>
        <CardHeader>
          <CardTitle>Upload a source video</CardTitle>
          <CardDescription>
            Supported formats: MP4, MOV, and AVI. Upload creates a queued
            analysis.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4">
          <div className="grid gap-2">
            <Label htmlFor="source-video">Video file</Label>
            <Input
              accept=".mp4,.mov,.avi,video/mp4,video/quicktime,video/x-msvideo"
              disabled={uploadMutation.isPending}
              id="source-video"
              onChange={(event) => {
                setSelectedFile(event.target.files?.[0] ?? null)
                setValidationError("")
                uploadMutation.reset()
              }}
              type="file"
            />
            {selectedFile && (
              <p className="text-sm text-muted-foreground">
                {selectedFile.name} ·{" "}
                {(selectedFile.size / (1024 * 1024)).toFixed(1)} MB
              </p>
            )}
          </div>
          {uploadProgress !== null && uploadMutation.isPending && (
            <div className="grid gap-1">
              <progress
                aria-label="Video upload progress"
                className="h-2 w-full accent-primary"
                max={100}
                value={uploadProgress}
              />
              <p className="text-sm text-muted-foreground">
                Uploading: {uploadProgress}%
              </p>
            </div>
          )}
          {validationError && (
            <p className="text-sm text-destructive" role="alert">
              {validationError}
            </p>
          )}
          {uploadMutation.isError && (
            <Alert variant="destructive">
              <AlertTitle>Upload failed</AlertTitle>
              <AlertDescription>
                {uploadMutation.error.message} Select the file again or retry.
              </AlertDescription>
            </Alert>
          )}
          <div>
            <Button
              disabled={uploadMutation.isPending}
              onClick={upload}
              type="button"
            >
              <Upload />
              {uploadMutation.isError ? "Retry upload" : "Upload and analyze"}
            </Button>
          </div>
        </CardContent>
      </Card>

      <section className="grid gap-4" aria-label="Source videos and analyses">
        <div>
          <h2 className="text-xl font-semibold">Source videos and analyses</h2>
          <p className="text-sm text-muted-foreground">
            Job status and frame progress are refreshed from the backend.
          </p>
        </div>
        {cancelMutation.isError && (
          <Alert variant="destructive">
            <AlertTitle>Cancellation failed</AlertTitle>
            <AlertDescription>{cancelMutation.error.message}</AlertDescription>
          </Alert>
        )}
        {analysesQuery.isPending ? (
          <p role="status">Loading analyses…</p>
        ) : analysesQuery.isError ? (
          <p className="text-destructive" role="alert">
            Analyses could not be loaded. {analysesQuery.error.message}
          </p>
        ) : analysesQuery.data.length === 0 ? (
          <p className="rounded-lg border border-dashed p-8 text-center text-muted-foreground">
            No source videos yet. Upload one above to create an analysis.
          </p>
        ) : (
          analysesQuery.data.map((analysis) => (
            <AnalysisCard
              analysis={analysis}
              key={analysis.id}
              onCancel={() => cancelMutation.mutate(analysis.id)}
              cancelling={cancelMutation.isPending}
            />
          ))
        )}
      </section>
    </div>
  )
}

function AnalysisCard({
  analysis,
  onCancel,
  cancelling,
}: {
  analysis: AnalysisPublic
  onCancel: () => void
  cancelling: boolean
}) {
  const percent =
    analysis.total_frames && analysis.total_frames > 0
      ? Math.min(
          100,
          Math.floor((analysis.frames_processed / analysis.total_frames) * 100),
        )
      : null

  return (
    <Card>
      <CardHeader className="flex flex-row items-start justify-between gap-4">
        <div className="grid gap-2">
          <CardTitle className="text-base">
            {analysis.video_original_filename}
          </CardTitle>
          <CardDescription>
            Added {new Date(analysis.created_at).toLocaleString()}
          </CardDescription>
        </div>
        <Badge variant={statusVariant(analysis.status)}>
          {analysis.status}
        </Badge>
      </CardHeader>
      <CardContent className="grid gap-3">
        {analysis.total_frames !== null && (
          <div className="grid gap-1">
            <div className="flex justify-between text-sm">
              <span>
                {analysis.frames_processed} / {analysis.total_frames} frames
              </span>
              <span>{percent === null ? "—" : `${percent}%`}</span>
            </div>
            <progress
              aria-label={`${analysis.video_original_filename} processing progress`}
              className="h-2 w-full accent-primary"
              max={analysis.total_frames || 1}
              value={analysis.frames_processed}
            />
          </div>
        )}
        <p className="text-sm text-muted-foreground">
          Elapsed: {formatElapsed(analysis.started_at, analysis.finished_at)}
        </p>
        {analysis.error_code && (
          <p className="text-sm text-destructive">
            {analysis.error_code}: {analysis.status}
          </p>
        )}
        {analysis.status === "succeeded" && (
          <Link
            className="inline-flex w-fit items-center gap-2 text-sm font-medium text-primary hover:underline"
            params={{ analysisId: analysis.id }}
            to="/analyses/$analysisId"
          >
            <ExternalLink className="size-4" />
            Open results
          </Link>
        )}
        {activeStatuses.includes(analysis.status) && (
          <div>
            <Button
              disabled={cancelling || analysis.status === "cancel_requested"}
              onClick={onCancel}
              size="sm"
              variant="outline"
            >
              <Ban />
              {analysis.status === "cancel_requested"
                ? "Cancellation requested"
                : "Cancel analysis"}
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
