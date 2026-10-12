import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { createFileRoute, Link, useNavigate } from "@tanstack/react-router"
import {
  ArrowLeft,
  ArrowRight,
  Circle,
  LoaderCircle,
  Pause,
  Pencil,
  Play,
  Plus,
  RectangleHorizontal,
  Save,
  SkipBack,
  SkipForward,
  Trash2,
  Triangle,
  Type,
  Undo2,
  Video,
} from "lucide-react"
import type { PointerEvent, MouseEvent as ReactMouseEvent } from "react"
import { useEffect, useMemo, useRef, useState } from "react"

import { client } from "@/client/client.gen"
import { ProjectCategories } from "@/components/Projects/ProjectCategories"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
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

type ClientProject = {
  id: string
  type: "media"
  kind: string
  workspace?: string
  title: string
  createdAt: string
  updatedAt: string
}

type Point = { x: number; y: number }
type ArrowVariant = "solid" | "curve" | "lob" | "marker" | "bounce"
type RingVariant =
  | "ground"
  | "dashed"
  | "radar"
  | "glow"
  | "shockwave"
  | "spotlight"
  | "team"
  | "pulse"
  | "radar-animated"
  | "glow-pulse"
type RectangleVariant =
  | "box"
  | "dash"
  | "hatch"
  | "hatch-horizontal"
  | "hatch-cross"
  | "square"
type TextVariant = "label" | "plain"
type AnnotationBase = {
  id: string
  videoId: string
  eventId?: string
  time: number
  duration: number
  color: string
  size: number
}
type Annotation =
  | (AnnotationBase & { type: "freehand"; points: Point[] })
  | (AnnotationBase & {
      type: "arrow"
      start: Point
      end: Point
      arrowVariant?: ArrowVariant
      dashed?: boolean
    })
  | (AnnotationBase & {
      type: "line"
      start: Point
      end: Point
      dashed?: boolean
    })
  | (AnnotationBase & {
      type: "circle"
      start: Point
      end: Point
      ringVariant?: RingVariant
    })
  | (AnnotationBase & {
      type: "rectangle"
      start: Point
      end: Point
      rectangleVariant?: RectangleVariant
    })
  | (AnnotationBase & { type: "funnel"; start: Point; end: Point })
  | (AnnotationBase & {
      type: "text"
      point: Point
      text: string
      textVariant?: TextVariant
    })
type FreezeMoment = {
  id: string
  videoId: string
  time: number
  duration: number
}
type EditorSnapshot = {
  annotations: Annotation[]
  moments: FreezeMoment[]
}
type DrawingTool =
  | "select"
  | "freehand"
  | "arrow"
  | "circle"
  | "rectangle"
  | "funnel"
  | "text"
  | "line"

type VideoAsset = {
  id: string
  filename: string
  media_type: string
  size_bytes: number
}

type ProjectPayload = Record<string, unknown> & {
  title?: string
  telestratorState?: { version?: number; drawings?: unknown; moments?: unknown }
}

type ProjectResponse = { payload: ProjectPayload; kind: string }
type ProjectListResponse = { projects: ClientProject[] }
type VideoListResponse = { videos: VideoAsset[] }

const projectPath = "/api/projects/media"
const playbackRefreshMs = 10 * 60 * 1000
const momentMatchToleranceSeconds = 0.4

export const Route = createFileRoute("/_layout/projects/telestrator")({
  component: TelestratorProjectsPage,
  validateSearch: (search: Record<string, unknown>) => ({
    projectId:
      typeof search.projectId === "string" ? search.projectId : undefined,
  }),
  head: () => ({
    meta: [{ title: "Telestrator | SokaFrame" }],
  }),
})

function projectPayload(
  id: string,
  title: string,
  timestamp: string,
): Record<string, unknown> {
  return {
    schemaVersion: 1,
    id,
    type: "media",
    kind: "tactical",
    workspace: "telestrator",
    title,
    createdAt: timestamp,
    updatedAt: timestamp,
    settings: {},
    boardState: { tokens: [] },
    mediaState: null,
    analysisState: null,
    telestratorState: { version: 1, drawings: [] },
  }
}

function authSecurity() {
  return [{ scheme: "bearer" as const, type: "http" as const }]
}

function errorMessage(error: unknown): string {
  return error instanceof Error
    ? error.message
    : "An unexpected error occurred."
}

function isPoint(value: unknown): value is Point {
  if (typeof value !== "object" || value === null) return false
  const point = value as Record<string, unknown>
  return (
    typeof point.x === "number" &&
    Number.isFinite(point.x) &&
    typeof point.y === "number" &&
    Number.isFinite(point.y)
  )
}

function isAnnotation(value: unknown): value is Annotation {
  if (typeof value !== "object" || value === null) return false
  const drawing = value as Record<string, unknown>
  if (
    typeof drawing.id !== "string" ||
    typeof drawing.videoId !== "string" ||
    typeof drawing.time !== "number" ||
    typeof drawing.duration !== "number" ||
    typeof drawing.color !== "string" ||
    typeof drawing.size !== "number"
  ) {
    return false
  }
  if (drawing.type === "freehand") {
    return Array.isArray(drawing.points) && drawing.points.every(isPoint)
  }
  if (drawing.type === "arrow" || drawing.type === "line") {
    return (
      isPoint(drawing.start) &&
      isPoint(drawing.end) &&
      (drawing.type !== "arrow" ||
        drawing.arrowVariant === undefined ||
        ["solid", "curve", "lob", "marker", "bounce"].includes(
          String(drawing.arrowVariant),
        )) &&
      (drawing.dashed === undefined || typeof drawing.dashed === "boolean")
    )
  }
  if (drawing.type === "circle") {
    return (
      isPoint(drawing.start) &&
      isPoint(drawing.end) &&
      (drawing.ringVariant === undefined ||
        [
          "ground",
          "dashed",
          "radar",
          "glow",
          "shockwave",
          "spotlight",
          "team",
          "pulse",
          "radar-animated",
          "glow-pulse",
        ].includes(String(drawing.ringVariant)))
    )
  }
  if (drawing.type === "rectangle" || drawing.type === "funnel") {
    return (
      isPoint(drawing.start) &&
      isPoint(drawing.end) &&
      (drawing.type !== "rectangle" ||
        drawing.rectangleVariant === undefined ||
        [
          "box",
          "dash",
          "hatch",
          "hatch-horizontal",
          "hatch-cross",
          "square",
        ].includes(String(drawing.rectangleVariant)))
    )
  }
  return (
    drawing.type === "text" &&
    isPoint(drawing.point) &&
    typeof drawing.text === "string" &&
    (drawing.textVariant === undefined ||
      ["label", "plain"].includes(String(drawing.textVariant)))
  )
}

function storedDrawings(payload: ProjectPayload): Annotation[] {
  const state = payload.telestratorState
  if (!state || !Array.isArray(state.drawings)) return []
  return state.drawings.filter(isAnnotation)
}

function storedMoments(payload: ProjectPayload): FreezeMoment[] {
  const state = payload.telestratorState
  if (!state || !Array.isArray(state.moments)) return []
  return state.moments.filter((value): value is FreezeMoment => {
    if (typeof value !== "object" || value === null) return false
    const moment = value as Record<string, unknown>
    return (
      typeof moment.id === "string" &&
      typeof moment.videoId === "string" &&
      typeof moment.time === "number" &&
      Number.isFinite(moment.time) &&
      typeof moment.duration === "number" &&
      Number.isFinite(moment.duration)
    )
  })
}

function formatTime(seconds: number): string {
  const safeSeconds = Number.isFinite(seconds) ? Math.max(0, seconds) : 0
  const minutes = Math.floor(safeSeconds / 60)
  const wholeSeconds = Math.floor(safeSeconds % 60)
  const centiseconds = Math.floor((safeSeconds % 1) * 100)
  return `${minutes.toString().padStart(2, "0")}:${wholeSeconds.toString().padStart(2, "0")}.${centiseconds.toString().padStart(2, "0")}`
}

function TelestratorProjectsPage() {
  const { projectId } = Route.useSearch()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [projectName, setProjectName] = useState("")
  const projectsQuery = useQuery({
    queryKey: ["telestrator-projects"],
    queryFn: async () => {
      const response = await client.get<{ "200": ProjectListResponse }>({
        url: projectPath,
        responseType: "json",
        security: authSecurity(),
      })
      if (!response.data) {
        throw new Error("The server returned no project list.")
      }
      return response.data.projects.filter(
        (project) =>
          project.kind === "tactical" && project.workspace === "telestrator",
      )
    },
  })

  const createProject = useMutation({
    mutationFn: async (title: string) => {
      const id = crypto.randomUUID()
      const timestamp = new Date().toISOString()
      await client.put({
        url: `${projectPath}/${encodeURIComponent(id)}`,
        body: { payload: projectPayload(id, title, timestamp) },
        responseType: "json",
        security: authSecurity(),
      })
      return {
        id,
        type: "media" as const,
        kind: "tactical",
        workspace: "telestrator",
        title,
        createdAt: timestamp,
        updatedAt: timestamp,
      }
    },
    onSuccess: async (project) => {
      setProjectName("")
      await queryClient.invalidateQueries({
        queryKey: ["telestrator-projects"],
      })
      await navigate({
        to: "/projects/telestrator",
        search: { projectId: project.id },
      })
    },
  })

  if (projectId) {
    const project = projectsQuery.data?.find((item) => item.id === projectId)
    if (projectsQuery.isPending) {
      return (
        <p
          className="flex items-center gap-2 text-sm text-muted-foreground"
          role="status"
        >
          <LoaderCircle className="size-4 animate-spin" />
          Loading Telestrator projects…
        </p>
      )
    }
    if (projectsQuery.isError) {
      return (
        <Alert variant="destructive">
          <AlertTitle>Projects unavailable</AlertTitle>
          <AlertDescription>
            {errorMessage(projectsQuery.error)}
          </AlertDescription>
        </Alert>
      )
    }
    if (project) {
      return <TelestratorEditor key={project.id} project={project} />
    }
  }

  return (
    <div className="flex flex-col gap-7">
      <header>
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-primary">
          Projects
        </p>
        <h1 className="mt-1 text-3xl font-bold tracking-tight sm:text-4xl">
          Telestrator
        </h1>
        <p className="mt-2 max-w-2xl text-sm text-muted-foreground sm:text-base">
          Upload match footage, pause on a moment, and save time-linked tactical
          annotations.
        </p>
      </header>

      <ProjectCategories active="telestrator" />

      {projectId ? (
        <Alert variant="destructive">
          <AlertTitle>Telestrator project unavailable</AlertTitle>
          <AlertDescription>
            This project could not be found. Choose another project or create a
            new one below.
          </AlertDescription>
        </Alert>
      ) : null}

      <Card className="border-border/80">
        <CardHeader>
          <CardTitle>Create a Telestrator project</CardTitle>
          <CardDescription>
            Create a workspace for a match, training session, or tactical
            review.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form
            className="flex flex-col gap-3 sm:flex-row sm:items-end"
            onSubmit={(event) => {
              event.preventDefault()
              const title = projectName.trim()
              if (title) createProject.mutate(title)
            }}
          >
            <div className="grid flex-1 gap-2">
              <Label htmlFor="telestrator-project-name">Project name</Label>
              <Input
                id="telestrator-project-name"
                maxLength={120}
                onChange={(event) => setProjectName(event.target.value)}
                placeholder="e.g. Matchday 12 — second-half review"
                value={projectName}
              />
            </div>
            <Button
              disabled={!projectName.trim() || createProject.isPending}
              type="submit"
            >
              {createProject.isPending ? (
                <LoaderCircle className="size-4 animate-spin" />
              ) : (
                <Plus className="size-4" />
              )}
              Create project
            </Button>
          </form>
          {createProject.isError ? (
            <Alert className="mt-4" variant="destructive">
              <AlertTitle>Project could not be created</AlertTitle>
              <AlertDescription>
                {errorMessage(createProject.error)}
              </AlertDescription>
            </Alert>
          ) : null}
        </CardContent>
      </Card>

      {projectsQuery.isPending ? (
        <p
          className="flex items-center gap-2 text-sm text-muted-foreground"
          role="status"
        >
          <LoaderCircle className="size-4 animate-spin" />
          Loading Telestrator projects…
        </p>
      ) : projectsQuery.isError ? (
        <Alert variant="destructive">
          <AlertTitle>Projects unavailable</AlertTitle>
          <AlertDescription>
            {errorMessage(projectsQuery.error)}
          </AlertDescription>
        </Alert>
      ) : projectsQuery.data.length ? (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {projectsQuery.data.map((project) => (
            <Card
              className="group border-border/80 transition-all hover:-translate-y-0.5 hover:border-primary/40 hover:shadow-lg hover:shadow-black/10"
              key={project.id}
            >
              <CardHeader className="flex-row items-start gap-3">
                <span className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-primary/12 text-primary">
                  <Video aria-hidden="true" className="size-5" />
                </span>
                <div className="min-w-0">
                  <CardTitle className="truncate text-base">
                    {project.title}
                  </CardTitle>
                  <CardDescription className="mt-1">
                    Updated {new Date(project.updatedAt).toLocaleDateString()}
                  </CardDescription>
                </div>
              </CardHeader>
              <CardContent>
                <Button
                  asChild
                  className="w-full justify-between"
                  variant="outline"
                >
                  <Link
                    search={{ projectId: project.id }}
                    to="/projects/telestrator"
                  >
                    Open Telestrator
                    <ArrowRight className="size-4 transition-transform group-hover:translate-x-0.5" />
                  </Link>
                </Button>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <Card>
          <CardHeader>
            <CardTitle>No Telestrator projects yet</CardTitle>
            <CardDescription>
              Create your first project to upload a video and start annotating.
            </CardDescription>
          </CardHeader>
        </Card>
      )}
    </div>
  )
}

function TelestratorEditor({ project }: { project: ClientProject }) {
  const queryClient = useQueryClient()
  const videoRef = useRef<HTMLVideoElement>(null)
  const captionObjectUrl = useRef("")
  const undoHistory = useRef<EditorSnapshot[]>([])
  const drawingRevision = useRef(0)
  const playedMomentIds = useRef(new Set<string>())
  const momentPlaybackTimer = useRef<number | undefined>(undefined)
  const editGesture = useRef<
    | {
        annotationId: string
        mode: "move" | "start" | "end"
        origin: Point
        snapshot: Annotation[]
        moved: boolean
      }
    | undefined
  >(undefined)
  const momentDrag = useRef<
    { id: string; pointerId: number; moved: boolean } | undefined
  >(undefined)
  const [annotations, setAnnotations] = useState<Annotation[]>([])
  const [moments, setMoments] = useState<FreezeMoment[]>([])
  const [annotationsLoaded, setAnnotationsLoaded] = useState(false)
  const [selectedVideoId, setSelectedVideoId] = useState("")
  const [editingMomentId, setEditingMomentId] = useState("")
  const [playbackMomentId, setPlaybackMomentId] = useState("")
  const [playback, setPlayback] = useState<{ videoId: string; url: string }>()
  const [playbackError, setPlaybackError] = useState("")
  const [uploadProgress, setUploadProgress] = useState("")
  const [captionTrack, setCaptionTrack] = useState<{
    videoId: string
    url: string
  }>()
  const [tool, setTool] = useState<DrawingTool>("select")
  const [arrowVariant, setArrowVariant] = useState<ArrowVariant>("solid")
  const [dashedArrow, setDashedArrow] = useState(false)
  const [ringVariant, setRingVariant] = useState<RingVariant>("ground")
  const [rectangleVariant, setRectangleVariant] =
    useState<RectangleVariant>("box")
  const [textVariant, setTextVariant] = useState<TextVariant>("label")
  const [color, setColor] = useState("#ff3b30")
  const [lineSize, setLineSize] = useState(5)
  const [annotationDuration, setAnnotationDuration] = useState(4)
  const [draft, setDraft] = useState<Annotation>()
  const [selectedAnnotationId, setSelectedAnnotationId] = useState("")
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [aspectRatio, setAspectRatio] = useState(16 / 9)
  const [isPlaying, setIsPlaying] = useState(false)
  const [muted, setMuted] = useState(false)
  const [volume, setVolume] = useState(1)
  const [dirty, setDirty] = useState(false)
  const uploadInput = useRef<HTMLInputElement>(null)
  const captionsInput = useRef<HTMLInputElement>(null)
  const videoQuery = useQuery({
    queryKey: ["telestrator-videos", project.id],
    queryFn: async () => {
      const response = await client.get<{ "200": VideoListResponse }>({
        url: `${projectPath}/${encodeURIComponent(project.id)}/videos`,
        responseType: "json",
        security: authSecurity(),
      })
      if (!response.data) {
        throw new Error("The server returned no video list.")
      }
      return response.data.videos
    },
  })
  const projectQuery = useQuery({
    queryKey: ["telestrator-project", project.id],
    queryFn: async () => {
      const response = await client.get<{ "200": ProjectResponse }>({
        url: `${projectPath}/${encodeURIComponent(project.id)}`,
        responseType: "json",
        security: authSecurity(),
      })
      if (!response.data) {
        throw new Error("The server returned no project data.")
      }
      return response.data.payload
    },
    refetchOnWindowFocus: false,
  })
  const videos = videoQuery.data ?? []
  const selectedVideo =
    videos.find((video) => video.id === selectedVideoId) ?? videos[0]
  const selectedMoments = useMemo(
    () =>
      moments
        .filter((moment) => moment.videoId === selectedVideo?.id)
        .sort((first, second) => first.time - second.time),
    [moments, selectedVideo?.id],
  )
  const activeMomentId = editingMomentId || playbackMomentId
  const visibleAnnotations = useMemo(
    () =>
      annotations.filter((annotation) => {
        if (annotation.videoId !== selectedVideo?.id) return false
        if (annotation.eventId) {
          const moment = moments.find((item) => item.id === annotation.eventId)
          return (
            moment !== undefined &&
            (moment.id === activeMomentId ||
              (currentTime >= moment.time &&
                currentTime <= moment.time + moment.duration))
          )
        }
        return (
          currentTime >= annotation.time - 0.04 &&
          currentTime <= annotation.time + annotation.duration
        )
      }),
    [activeMomentId, annotations, currentTime, moments, selectedVideo?.id],
  )
  const viewBoxHeight = 1000 / aspectRatio

  useEffect(
    () => () => {
      if (captionObjectUrl.current) {
        URL.revokeObjectURL(captionObjectUrl.current)
      }
    },
    [],
  )

  useEffect(() => {
    if (!projectQuery.data || annotationsLoaded) return
    setAnnotations(storedDrawings(projectQuery.data))
    setMoments(storedMoments(projectQuery.data))
    setAnnotationsLoaded(true)
  }, [annotationsLoaded, projectQuery.data])

  useEffect(() => {
    if (!selectedVideo && videos[0]) {
      setSelectedVideoId(videos[0].id)
    }
  }, [selectedVideo, videos])

  useEffect(() => {
    if (!selectedVideo) {
      setPlayback(undefined)
      return
    }
    let cancelled = false
    const renewPlayback = async () => {
      try {
        const response = await client.post<{ "200": { url: string } }>({
          url: `${projectPath}/${encodeURIComponent(project.id)}/videos/${encodeURIComponent(selectedVideo.id)}/playback`,
          responseType: "json",
          security: authSecurity(),
        })
        if (!response.data?.url) {
          throw new Error("The server did not return a video playback URL.")
        }
        if (!cancelled) {
          setPlayback({ videoId: selectedVideo.id, url: response.data.url })
          setPlaybackError("")
        }
      } catch (error) {
        if (!cancelled) setPlaybackError(errorMessage(error))
      }
    }

    void renewPlayback()
    const intervalId = window.setInterval(
      () => void renewPlayback(),
      playbackRefreshMs,
    )
    const refreshOnFocus = () => {
      if (document.visibilityState === "visible") void renewPlayback()
    }
    document.addEventListener("visibilitychange", refreshOnFocus)
    return () => {
      cancelled = true
      window.clearInterval(intervalId)
      document.removeEventListener("visibilitychange", refreshOnFocus)
    }
  }, [project.id, selectedVideo])

  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    if (playback?.videoId !== selectedVideo?.id) {
      video.removeAttribute("src")
      video.load()
      return
    }
    video.src = playback.url
    video.load()
  }, [playback, selectedVideo?.id])

  const uploadVideo = useMutation({
    mutationFn: async (files: File[]) => {
      const uploadedVideos: VideoAsset[] = []
      for (const [index, file] of files.entries()) {
        setUploadProgress(
          `Uploading ${index + 1} of ${files.length}: ${file.name}`,
        )
        const form = new FormData()
        form.set("video_file", file)
        try {
          const response = await client.post<{ "201": VideoAsset }>({
            url: `${projectPath}/${encodeURIComponent(project.id)}/videos`,
            body: form,
            responseType: "json",
            security: authSecurity(),
          })
          if (!response.data) {
            throw new Error("The server returned no uploaded video details.")
          }
          uploadedVideos.push(response.data)
        } catch (error) {
          throw new Error(
            `Upload failed for ${file.name}: ${errorMessage(error)}`,
          )
        }
      }
      return uploadedVideos
    },
    onSuccess: async (uploadedVideos) => {
      const lastVideo = uploadedVideos[uploadedVideos.length - 1]
      if (lastVideo) setSelectedVideoId(lastVideo.id)
    },
    onSettled: async () => {
      setUploadProgress("")
      await queryClient.invalidateQueries({
        queryKey: ["telestrator-videos", project.id],
      })
    },
  })

  const saveAnnotations = useMutation({
    mutationFn: async (input: {
      drawings: Annotation[]
      moments: FreezeMoment[]
      revision: number
    }) => {
      if (!projectQuery.data) {
        throw new Error("Project data has not finished loading.")
      }
      const timestamp = new Date().toISOString()
      const payload = {
        ...projectQuery.data,
        updatedAt: timestamp,
        telestratorState: {
          version: 1,
          drawings: input.drawings,
          moments: input.moments,
        },
      }
      await client.put({
        url: `${projectPath}/${encodeURIComponent(project.id)}`,
        body: { payload },
        responseType: "json",
        security: authSecurity(),
      })
      return input.revision
    },
    onSuccess: async (savedRevision) => {
      if (savedRevision === drawingRevision.current) setDirty(false)
      await queryClient.invalidateQueries({
        queryKey: ["telestrator-projects"],
      })
    },
  })

  const addHistory = () => {
    undoHistory.current = [
      ...undoHistory.current.slice(-49),
      { annotations, moments },
    ]
  }

  const addHistorySnapshot = (snapshot: EditorSnapshot) => {
    undoHistory.current = [...undoHistory.current.slice(-49), snapshot]
  }

  const markDirty = () => {
    drawingRevision.current += 1
    setDirty(true)
  }

  const pointFromPointer = (event: PointerEvent<SVGSVGElement>): Point => {
    const bounds = event.currentTarget.getBoundingClientRect()
    return {
      x: Math.min(1, Math.max(0, (event.clientX - bounds.left) / bounds.width)),
      y: Math.min(1, Math.max(0, (event.clientY - bounds.top) / bounds.height)),
    }
  }

  const pointDelta = (first: Point, second: Point): Point => ({
    x: second.x - first.x,
    y: second.y - first.y,
  })

  const shiftAnnotation = (
    annotation: Annotation,
    delta: Point,
  ): Annotation => {
    const shift = (point: Point): Point => ({
      x: Math.min(1, Math.max(0, point.x + delta.x)),
      y: Math.min(1, Math.max(0, point.y + delta.y)),
    })
    if (annotation.type === "freehand") {
      return { ...annotation, points: annotation.points.map(shift) }
    }
    if (annotation.type === "text") {
      return { ...annotation, point: shift(annotation.point) }
    }
    return {
      ...annotation,
      start: shift(annotation.start),
      end: shift(annotation.end),
    }
  }

  const resizeAnnotation = (
    annotation: Annotation,
    mode: "start" | "end",
    point: Point,
  ): Annotation => {
    if (annotation.type === "text" || annotation.type === "freehand") {
      return annotation
    }
    if (mode === "start") return { ...annotation, start: point }
    return { ...annotation, end: point }
  }

  const onDrawingStart = (event: PointerEvent<SVGSVGElement>) => {
    if (!selectedVideo) return
    const target = event.target
    const svgTarget = target instanceof SVGElement ? target : undefined
    const targetNode = svgTarget?.closest<SVGElement>("[data-annotation-id]")
    const targetId = targetNode?.dataset.annotationId
    const editHandle = svgTarget
      ?.closest<SVGElement>("[data-edit-handle]")
      ?.getAttribute("data-edit-handle")
    if (tool === "select") {
      setSelectedAnnotationId(targetId ?? "")
      if (!targetId) {
        editGesture.current = undefined
        return
      }
      const mode =
        editHandle === "start" || editHandle === "end" ? editHandle : "move"
      editGesture.current = {
        annotationId: targetId,
        mode,
        origin: pointFromPointer(event),
        snapshot: annotations,
        moved: false,
      }
      event.currentTarget.setPointerCapture(event.pointerId)
      return
    }
    const point = pointFromPointer(event)
    if (tool === "text") {
      videoRef.current?.pause()
      setIsPlaying(false)
      const text = window.prompt("Annotation text")
      if (!text?.trim()) return
      const next: Annotation = {
        id: crypto.randomUUID(),
        videoId: selectedVideo.id,
        ...(editingMomentId ? { eventId: editingMomentId } : {}),
        time:
          moments.find((moment) => moment.id === editingMomentId)?.time ??
          videoRef.current?.currentTime ??
          currentTime,
        duration:
          moments.find((moment) => moment.id === editingMomentId)?.duration ??
          annotationDuration,
        color,
        size: lineSize,
        type: "text",
        point,
        text: text.trim(),
        textVariant,
      }
      addHistory()
      setAnnotations((current) => [...current, next])
      markDirty()
      return
    }
    videoRef.current?.pause()
    setIsPlaying(false)
    setSelectedAnnotationId("")
    const base: AnnotationBase = {
      id: crypto.randomUUID(),
      videoId: selectedVideo.id,
      ...(editingMomentId ? { eventId: editingMomentId } : {}),
      time:
        moments.find((moment) => moment.id === editingMomentId)?.time ??
        videoRef.current?.currentTime ??
        currentTime,
      duration:
        moments.find((moment) => moment.id === editingMomentId)?.duration ??
        annotationDuration,
      color,
      size: lineSize,
    }
    if (tool === "freehand") {
      setDraft({ ...base, type: "freehand", points: [point] })
    } else if (tool === "arrow") {
      setDraft({
        ...base,
        type: "arrow",
        start: point,
        end: point,
        arrowVariant,
        dashed: dashedArrow,
      })
    } else if (tool === "line") {
      setDraft({
        ...base,
        type: "line",
        start: point,
        end: point,
        dashed: dashedArrow,
      })
    } else if (tool === "circle") {
      setDraft({
        ...base,
        type: "circle",
        start: point,
        end: point,
        ringVariant,
      })
    } else if (tool === "funnel") {
      setDraft({ ...base, type: "funnel", start: point, end: point })
    } else {
      setDraft({
        ...base,
        type: "rectangle",
        start: point,
        end: point,
        rectangleVariant,
      })
    }
    event.currentTarget.setPointerCapture(event.pointerId)
  }

  const onDrawingMove = (event: PointerEvent<SVGSVGElement>) => {
    const gesture = editGesture.current
    if (gesture) {
      const point = pointFromPointer(event)
      const delta = pointDelta(gesture.origin, point)
      if (Math.abs(delta.x) + Math.abs(delta.y) < 0.0005) return
      gesture.moved = true
      const original = gesture.snapshot.find(
        (annotation) => annotation.id === gesture.annotationId,
      )
      if (!original) return
      setAnnotations((current) =>
        current.map((annotation) => {
          if (annotation.id !== gesture.annotationId) return annotation
          if (gesture.mode === "move") return shiftAnnotation(original, delta)
          return resizeAnnotation(original, gesture.mode, point)
        }),
      )
      return
    }
    if (!draft) return
    let point = pointFromPointer(event)
    if (draft.type === "freehand") {
      setDraft({ ...draft, points: [...draft.points, point] })
    } else if (draft.type !== "text") {
      if (draft.type === "rectangle" && draft.rectangleVariant === "square") {
        const horizontal = point.x - draft.start.x
        const vertical = point.y - draft.start.y
        if (Math.abs(horizontal) >= Math.abs(vertical / aspectRatio)) {
          point = {
            ...point,
            y: Math.min(
              1,
              Math.max(
                0,
                draft.start.y +
                  Math.sign(vertical || 1) * Math.abs(horizontal) * aspectRatio,
              ),
            ),
          }
        } else {
          point = {
            ...point,
            x: Math.min(
              1,
              Math.max(
                0,
                draft.start.x +
                  Math.sign(horizontal || 1) * Math.abs(vertical / aspectRatio),
              ),
            ),
          }
        }
      }
      setDraft({ ...draft, end: point })
    }
  }

  const onDrawingEnd = () => {
    const gesture = editGesture.current
    if (gesture) {
      editGesture.current = undefined
      if (gesture.moved) {
        addHistorySnapshot({ annotations: gesture.snapshot, moments })
        markDirty()
      }
      return
    }
    if (!draft) return
    addHistory()
    setAnnotations((current) => [...current, draft])
    setDraft(undefined)
    markDirty()
  }

  const undo = () => {
    const previous = undoHistory.current[undoHistory.current.length - 1]
    if (!previous) return
    undoHistory.current = undoHistory.current.slice(0, -1)
    setAnnotations(previous.annotations)
    setMoments(previous.moments)
    markDirty()
  }

  const removeSelected = () => {
    if (!selectedAnnotationId) return
    addHistory()
    setAnnotations((current) =>
      current.filter((annotation) => annotation.id !== selectedAnnotationId),
    )
    setSelectedAnnotationId("")
    markDirty()
  }

  const clearAnnotations = () => {
    if (
      !selectedVideo ||
      !annotations.some((item) => item.videoId === selectedVideo.id)
    ) {
      return
    }
    addHistory()
    setAnnotations((current) =>
      current.filter((annotation) => annotation.videoId !== selectedVideo.id),
    )
    setSelectedAnnotationId("")
    markDirty()
  }

  const openMoment = (moment: FreezeMoment) => {
    const video = videoRef.current
    if (!video) return
    video.pause()
    setPlaybackMomentId("")
    setEditingMomentId(moment.id)
    setAnnotationDuration(moment.duration)
    if (Math.abs(video.currentTime - moment.time) > 0.02) {
      video.currentTime = moment.time
    }
  }

  const startFreezeMoment = () => {
    if (!selectedVideo) return
    const video = videoRef.current
    if (!video) return
    video.pause()
    if (momentPlaybackTimer.current !== undefined) {
      window.clearTimeout(momentPlaybackTimer.current)
      momentPlaybackTimer.current = undefined
    }
    setPlaybackMomentId("")
    const time = Math.round(video.currentTime * 100) / 100
    const existing = selectedMoments.find(
      (moment) => Math.abs(moment.time - time) <= momentMatchToleranceSeconds,
    )
    if (existing) {
      openMoment(existing)
      return
    }
    const moment: FreezeMoment = {
      id: crypto.randomUUID(),
      videoId: selectedVideo.id,
      time,
      duration: 4,
    }
    setMoments((current) => [...current, moment])
    setEditingMomentId(moment.id)
    setAnnotationDuration(moment.duration)
    markDirty()
  }

  const finishFreezeMoment = (resume: boolean) => {
    const momentId = editingMomentId
    setEditingMomentId("")
    if (momentId) {
      playedMomentIds.current.add(momentId)
      if (resume) {
        void videoRef.current?.play().catch((error: unknown) => {
          setPlaybackError(errorMessage(error))
        })
      }
    }
  }

  const deleteFreezeMoment = () => {
    if (!editingMomentId) return
    addHistory()
    setMoments((current) =>
      current.filter((moment) => moment.id !== editingMomentId),
    )
    setAnnotations((current) =>
      current.filter((annotation) => annotation.eventId !== editingMomentId),
    )
    setSelectedAnnotationId("")
    setEditingMomentId("")
    markDirty()
  }

  const updateFreezeDuration = (value: number) => {
    const nextDuration = Math.min(30, Math.max(1, Math.round(value)))
    setAnnotationDuration(nextDuration)
    if (!editingMomentId) return
    setMoments((current) =>
      current.map((moment) =>
        moment.id === editingMomentId
          ? { ...moment, duration: nextDuration }
          : moment,
      ),
    )
    setAnnotations((current) =>
      current.map((annotation) =>
        annotation.eventId === editingMomentId
          ? { ...annotation, duration: nextDuration }
          : annotation,
      ),
    )
    markDirty()
  }

  const handleVideoTimeUpdate = (video: HTMLVideoElement) => {
    setCurrentTime(video.currentTime)
    if (video.paused || editingMomentId || playbackMomentId) return
    const moment = selectedMoments.find(
      (item) =>
        item.time <= video.currentTime + 0.04 &&
        !playedMomentIds.current.has(item.id),
    )
    if (!moment) return
    playedMomentIds.current.add(moment.id)
    setPlaybackMomentId(moment.id)
    video.pause()
    momentPlaybackTimer.current = window.setTimeout(() => {
      momentPlaybackTimer.current = undefined
      setPlaybackMomentId("")
      if (videoRef.current === video) {
        void video.play().catch((error: unknown) => {
          setPlaybackError(errorMessage(error))
        })
      }
    }, moment.duration * 1000)
  }

  const handleVideoSeeking = (video: HTMLVideoElement) => {
    selectedMoments.forEach((moment) => {
      if (moment.time < video.currentTime - 0.15) {
        playedMomentIds.current.add(moment.id)
      } else {
        playedMomentIds.current.delete(moment.id)
      }
    })
    const activeMoment = moments.find(
      (moment) =>
        moment.id === editingMomentId || moment.id === playbackMomentId,
    )
    if (activeMoment && Math.abs(video.currentTime - activeMoment.time) > 0.5) {
      if (momentPlaybackTimer.current !== undefined) {
        window.clearTimeout(momentPlaybackTimer.current)
        momentPlaybackTimer.current = undefined
      }
      setEditingMomentId("")
      setPlaybackMomentId("")
    }
  }

  const finishMomentDrag = (event: PointerEvent<HTMLFieldSetElement>) => {
    const drag = momentDrag.current
    if (!drag || drag.pointerId !== event.pointerId) return
    momentDrag.current = undefined
    if (!drag.moved) {
      const moment = selectedMoments.find((item) => item.id === drag.id)
      if (moment) openMoment(moment)
      return
    }
    const bounds = event.currentTarget.getBoundingClientRect()
    const ratio = Math.min(
      1,
      Math.max(0, (event.clientX - bounds.left) / bounds.width),
    )
    const time = Math.round(ratio * duration * 100) / 100
    setMoments((current) =>
      current.map((moment) =>
        moment.id === drag.id ? { ...moment, time } : moment,
      ),
    )
    setAnnotations((current) =>
      current.map((annotation) =>
        annotation.eventId === drag.id ? { ...annotation, time } : annotation,
      ),
    )
    if (editingMomentId === drag.id) {
      if (videoRef.current) videoRef.current.currentTime = time
    }
    markDirty()
  }

  const onTextDoubleClick = (event: ReactMouseEvent<SVGSVGElement>) => {
    const target = event.target
    const node =
      target instanceof SVGElement
        ? target.closest<SVGElement>("[data-annotation-id]")
        : undefined
    const annotation = annotations.find(
      (item) => item.id === node?.dataset.annotationId && item.type === "text",
    )
    if (annotation?.type !== "text") return
    const text = window.prompt("Edit annotation text", annotation.text)
    if (text === null || !text.trim()) return
    addHistory()
    setAnnotations((current) =>
      current.map((item) =>
        item.id === annotation.id && item.type === "text"
          ? { ...item, text: text.trim() }
          : item,
      ),
    )
    markDirty()
  }

  const seek = (time: number) => {
    const video = videoRef.current
    if (!video) return
    const nextTime = Math.min(
      Math.max(time, 0),
      duration || Number.MAX_SAFE_INTEGER,
    )
    if (editingMomentId) {
      const moment = moments.find((item) => item.id === editingMomentId)
      if (!moment || Math.abs(moment.time - nextTime) > 0.5) {
        setEditingMomentId("")
      }
    }
    video.currentTime = nextTime
  }

  const togglePlayback = () => {
    const video = videoRef.current
    if (!video) return
    if (editingMomentId) {
      finishFreezeMoment(false)
    }
    if (video.paused) {
      void video.play().catch((error: unknown) => {
        setPlaybackError(errorMessage(error))
      })
    } else {
      video.pause()
    }
  }

  useEffect(() => {
    if (!selectedVideo?.id) return
    playedMomentIds.current.clear()
    setEditingMomentId("")
    setPlaybackMomentId("")
    if (momentPlaybackTimer.current !== undefined) {
      window.clearTimeout(momentPlaybackTimer.current)
      momentPlaybackTimer.current = undefined
    }
    return () => {
      if (momentPlaybackTimer.current !== undefined) {
        window.clearTimeout(momentPlaybackTimer.current)
        momentPlaybackTimer.current = undefined
      }
    }
  }, [selectedVideo?.id])

  const keyboardActions = useRef({
    togglePlayback,
    seek,
    undo,
    removeSelected,
    hasSelection: () => selectedAnnotationId.length > 0,
  })
  keyboardActions.current = {
    togglePlayback,
    seek,
    undo,
    removeSelected,
    hasSelection: () => selectedAnnotationId.length > 0,
  }

  useEffect(() => {
    const handleKeyboard = (event: KeyboardEvent) => {
      const target = event.target
      if (
        target instanceof HTMLElement &&
        (target.isContentEditable ||
          ["INPUT", "TEXTAREA", "SELECT", "BUTTON"].includes(target.tagName))
      ) {
        return
      }
      if (event.code === "Space") {
        event.preventDefault()
        keyboardActions.current.togglePlayback()
      } else if (event.key === "ArrowLeft") {
        event.preventDefault()
        keyboardActions.current.seek(
          (videoRef.current?.currentTime ?? 0) - 1 / 25,
        )
      } else if (event.key === "ArrowRight") {
        event.preventDefault()
        keyboardActions.current.seek(
          (videoRef.current?.currentTime ?? 0) + 1 / 25,
        )
      } else if (
        (event.ctrlKey || event.metaKey) &&
        event.key.toLowerCase() === "z"
      ) {
        event.preventDefault()
        keyboardActions.current.undo()
      } else if (
        (event.key === "Delete" || event.key === "Backspace") &&
        keyboardActions.current.hasSelection()
      ) {
        event.preventDefault()
        keyboardActions.current.removeSelected()
      }
    }
    window.addEventListener("keydown", handleKeyboard)
    return () => window.removeEventListener("keydown", handleKeyboard)
  }, [])

  if (projectQuery.isPending || videoQuery.isPending) {
    return (
      <p
        className="flex items-center gap-2 text-sm text-muted-foreground"
        role="status"
      >
        <LoaderCircle className="size-4 animate-spin" />
        Loading Telestrator…
      </p>
    )
  }
  if (projectQuery.isError || videoQuery.isError) {
    return (
      <Alert variant="destructive">
        <AlertTitle>Telestrator could not be loaded</AlertTitle>
        <AlertDescription>
          {errorMessage(projectQuery.error ?? videoQuery.error)}
        </AlertDescription>
      </Alert>
    )
  }

  const renderAnnotation = (annotation: Annotation) => {
    const selected = selectedAnnotationId === annotation.id
    const strokeWidth = annotation.size + (selected ? 2 : 0)
    const x = (point: Point) => point.x * 1000
    const y = (point: Point) => point.y * viewBoxHeight
    const renderFunnel = (start: Point, end: Point, id: string) => {
      const startX = x(start)
      const startY = y(start)
      const endX = x(end)
      const endY = y(end)
      const deltaX = endX - startX
      const deltaY = endY - startY
      const length = Math.hypot(deltaX, deltaY) || 1
      const radius = Math.max(14, length * 0.28)
      const offsetX = (-deltaY / length) * radius
      const offsetY = (deltaX / length) * radius
      return (
        <g data-annotation-id={id} key={id}>
          <path
            d={`M ${startX} ${startY} L ${endX + offsetX} ${endY + offsetY} Q ${endX} ${endY + radius * 0.28} ${endX - offsetX} ${endY - offsetY} Z`}
            fill={`${annotation.color}30`}
            stroke={annotation.color}
            strokeWidth={strokeWidth}
          />
          <ellipse
            cx={endX}
            cy={endY}
            rx={radius}
            ry={Math.max(7, radius * 0.28)}
            fill={`${annotation.color}20`}
            stroke={annotation.color}
            strokeWidth={strokeWidth}
          />
        </g>
      )
    }
    if (annotation.type === "freehand") {
      return (
        <polyline
          data-annotation-id={annotation.id}
          fill="none"
          key={annotation.id}
          points={annotation.points
            .map((point) => `${x(point)},${y(point)}`)
            .join(" ")}
          stroke={annotation.color}
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={strokeWidth}
        />
      )
    }
    if (annotation.type === "arrow" || annotation.type === "line") {
      const variant =
        annotation.type === "arrow"
          ? (annotation.arrowVariant ?? "solid")
          : "solid"
      const startX = x(annotation.start)
      const startY = y(annotation.start)
      const endX = x(annotation.end)
      const endY = y(annotation.end)
      const length = Math.hypot(endX - startX, endY - startY)
      const deltaX = endX - startX
      const deltaY = endY - startY
      const controlX =
        (startX + endX) / 2 - (deltaY / (length || 1)) * length * 0.2
      const controlY =
        variant === "lob"
          ? Math.max(16, (startY + endY) / 2 - Math.max(28, length * 0.42))
          : (startY + endY) / 2 +
            (deltaX / (length || 1)) * Math.max(12, Math.min(100, length * 0.2))
      const path =
        annotation.type === "arrow" &&
        (variant === "curve" || variant === "lob")
          ? `M ${startX} ${startY} Q ${controlX} ${variant === "lob" ? controlY : controlY} ${endX} ${endY}`
          : `M ${startX} ${startY} L ${endX} ${endY}`
      return (
        <path
          data-annotation-id={annotation.id}
          key={annotation.id}
          markerEnd={
            annotation.type === "arrow"
              ? `url(#telestrator-arrow-${annotation.id})`
              : undefined
          }
          d={path}
          fill="none"
          stroke={annotation.color}
          strokeDasharray={
            annotation.dashed || variant === "marker" || variant === "bounce"
              ? "13 9"
              : undefined
          }
          strokeLinecap="round"
          strokeWidth={strokeWidth}
        >
          {variant === "bounce" ? (
            <animate
              attributeName="stroke-dashoffset"
              dur="0.7s"
              from="0"
              repeatCount="indefinite"
              to="-20"
            />
          ) : null}
        </path>
      )
    }
    if (annotation.type === "circle") {
      const centerX = (x(annotation.start) + x(annotation.end)) / 2
      const centerY = (y(annotation.start) + y(annotation.end)) / 2
      const radiusX = Math.abs(x(annotation.end) - x(annotation.start)) / 2
      const radiusY = Math.abs(y(annotation.end) - y(annotation.start)) / 2
      const variant = annotation.ringVariant ?? "ground"
      const rings =
        variant === "shockwave" || variant === "radar-animated"
          ? [1, 0.72, 0.44]
          : variant === "team"
            ? [0.34, 0.34, 0.34]
            : [1]
      return (
        <g data-annotation-id={annotation.id} key={annotation.id}>
          {variant === "team"
            ? rings.map((scale, index) => (
                <ellipse
                  cx={centerX + (index - 1) * radiusX * 0.48}
                  cy={centerY}
                  key={index}
                  rx={Math.max(2, radiusX * scale)}
                  ry={Math.max(2, radiusY * scale)}
                  fill={`${annotation.color}18`}
                  stroke={annotation.color}
                  strokeWidth={strokeWidth}
                />
              ))
            : rings.map((scale, index) => (
                <ellipse
                  cx={centerX}
                  cy={centerY}
                  key={index}
                  rx={Math.max(2, radiusX * scale)}
                  ry={Math.max(2, radiusY * scale)}
                  fill={
                    variant === "ground"
                      ? `${annotation.color}16`
                      : variant === "glow" || variant === "glow-pulse"
                        ? `${annotation.color}08`
                        : "none"
                  }
                  opacity={
                    variant === "shockwave" ? 1 - index * 0.2 : undefined
                  }
                  stroke={annotation.color}
                  strokeDasharray={
                    variant === "dashed" || variant === "radar"
                      ? "9 7"
                      : undefined
                  }
                  strokeWidth={
                    variant === "glow" || variant === "glow-pulse"
                      ? strokeWidth + 3
                      : strokeWidth
                  }
                >
                  {variant === "pulse" || variant === "glow-pulse" ? (
                    <animate
                      attributeName="opacity"
                      dur="1.6s"
                      from="0.35"
                      repeatCount="indefinite"
                      to="1"
                    />
                  ) : variant === "radar-animated" ? (
                    <animate
                      attributeName="opacity"
                      dur="1.4s"
                      from={index === 0 ? "1" : "0.25"}
                      repeatCount="indefinite"
                      to={index === 0 ? "0.2" : "1"}
                    />
                  ) : null}
                </ellipse>
              ))}
          {variant === "radar" || variant === "radar-animated" ? (
            <path
              d={`M ${centerX} ${centerY} L ${centerX + radiusX} ${centerY}`}
              stroke={annotation.color}
              strokeWidth={strokeWidth}
            />
          ) : null}
        </g>
      )
    }
    if (annotation.type === "rectangle") {
      const rectVariant = annotation.rectangleVariant ?? "box"
      const rawWidth = Math.abs(x(annotation.end) - x(annotation.start))
      const rawHeight = Math.abs(y(annotation.end) - y(annotation.start))
      const squareSide = Math.min(rawWidth, rawHeight)
      const rectWidth = rectVariant === "square" ? squareSide : rawWidth
      const rectHeight = rectVariant === "square" ? squareSide : rawHeight
      return (
        <rect
          data-annotation-id={annotation.id}
          height={rectHeight}
          key={annotation.id}
          width={rectWidth}
          x={
            rectVariant === "square"
              ? x(annotation.start) <= x(annotation.end)
                ? x(annotation.start)
                : x(annotation.end)
              : Math.min(x(annotation.start), x(annotation.end))
          }
          y={
            rectVariant === "square"
              ? y(annotation.start) <= y(annotation.end)
                ? y(annotation.start)
                : y(annotation.end)
              : Math.min(y(annotation.start), y(annotation.end))
          }
          fill={
            rectVariant === "hatch" ||
            rectVariant === "hatch-horizontal" ||
            rectVariant === "hatch-cross"
              ? `url(#telestrator-pattern-${annotation.id})`
              : rectVariant === "dash"
                ? "none"
                : `${annotation.color}22`
          }
          strokeDasharray={rectVariant === "dash" ? "14 9" : undefined}
          stroke={annotation.color}
          strokeWidth={strokeWidth}
        />
      )
    }
    if (annotation.type === "funnel") {
      return renderFunnel(annotation.start, annotation.end, annotation.id)
    }
    const fontSize = Math.max(18, annotation.size * 5)
    const textWidth = Math.max(
      fontSize,
      annotation.text.length * fontSize * 0.58,
    )
    return (
      <g data-annotation-id={annotation.id} key={annotation.id}>
        {annotation.textVariant !== "plain" ? (
          <rect
            fill="#101827"
            fillOpacity="0.82"
            height={fontSize * 1.45}
            rx={fontSize * 0.14}
            width={textWidth + fontSize * 0.6}
            x={x(annotation.point)}
            y={y(annotation.point) - fontSize * 1.12}
          />
        ) : null}
        <text
          fill={annotation.color}
          fontSize={fontSize}
          fontWeight="bold"
          stroke={annotation.textVariant === "plain" ? "#111827" : "none"}
          strokeWidth={annotation.textVariant === "plain" ? 2 : 0}
          paintOrder="stroke"
          x={x(annotation.point) + fontSize * 0.3}
          y={y(annotation.point)}
        >
          {annotation.text}
        </text>
      </g>
    )
  }

  const projectedDrawings = draft
    ? [...visibleAnnotations, draft]
    : visibleAnnotations
  const renderEditHandles = (annotation: Annotation) => {
    if (annotation.id !== selectedAnnotationId || annotation.id === draft?.id) {
      return null
    }
    const handle = (point: Point, mode: "move" | "start" | "end") => (
      <circle
        cx={point.x * 1000}
        cy={point.y * viewBoxHeight}
        data-annotation-id={annotation.id}
        data-edit-handle={mode}
        fill="#fff"
        key={`${annotation.id}-${mode}`}
        r={12}
        stroke="#0f172a"
        strokeWidth={3}
      />
    )
    if (annotation.type === "text") return handle(annotation.point, "move")
    if (annotation.type === "freehand") {
      const xs = annotation.points.map((point) => point.x)
      const ys = annotation.points.map((point) => point.y)
      return handle(
        {
          x: (Math.min(...xs) + Math.max(...xs)) / 2,
          y: (Math.min(...ys) + Math.max(...ys)) / 2,
        },
        "move",
      )
    }
    return (
      <g key={annotation.id}>
        {handle(annotation.start, "start")}
        {handle(annotation.end, "end")}
      </g>
    )
  }

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <Link
            className="mb-3 inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground"
            search={{ projectId: undefined }}
            to="/projects/telestrator"
          >
            <ArrowLeft className="size-4" />
            All Telestrator projects
          </Link>
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-primary">
            Telestrator
          </p>
          <h1 className="mt-1 text-3xl font-bold tracking-tight sm:text-4xl">
            {project.title}
          </h1>
          <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
            Pause the footage to draw. Annotations stay attached to their video
            and timestamp.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">
            {dirty ? "Unsaved changes" : "All changes saved"}
          </span>
          <Button
            disabled={!dirty || !annotationsLoaded || saveAnnotations.isPending}
            onClick={() =>
              saveAnnotations.mutate({
                drawings: annotations,
                moments,
                revision: drawingRevision.current,
              })
            }
          >
            {saveAnnotations.isPending ? (
              <LoaderCircle className="size-4 animate-spin" />
            ) : (
              <Save className="size-4" />
            )}
            Save annotations
          </Button>
        </div>
      </header>

      <ProjectCategories active="telestrator" />

      {saveAnnotations.isError ? (
        <Alert variant="destructive">
          <AlertTitle>Annotations could not be saved</AlertTitle>
          <AlertDescription>
            {errorMessage(saveAnnotations.error)}
          </AlertDescription>
        </Alert>
      ) : null}
      {uploadVideo.isError ? (
        <Alert variant="destructive">
          <AlertTitle>Video could not be uploaded</AlertTitle>
          <AlertDescription>{errorMessage(uploadVideo.error)}</AlertDescription>
        </Alert>
      ) : null}
      {uploadProgress ? (
        <p className="text-sm text-muted-foreground" role="status">
          {uploadProgress}
        </p>
      ) : null}

      <div className="grid items-start gap-5 xl:grid-cols-[minmax(0,1fr)_19rem]">
        <Card className="min-w-0 border-border/80">
          <CardHeader>
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <CardTitle>Match footage</CardTitle>
                <CardDescription className="mt-1">
                  Supported formats: MP4, MOV, AVI. Maximum upload size is set
                  by the server.
                </CardDescription>
              </div>
              <Button
                disabled={uploadVideo.isPending}
                onClick={() => uploadInput.current?.click()}
                variant="outline"
              >
                {uploadVideo.isPending ? (
                  <LoaderCircle className="size-4 animate-spin" />
                ) : (
                  <Plus className="size-4" />
                )}
                Upload video
              </Button>
              {selectedVideo ? (
                <Button
                  onClick={() => captionsInput.current?.click()}
                  variant="outline"
                >
                  Load captions
                </Button>
              ) : null}
              <Input
                accept="video/mp4,video/quicktime,video/x-msvideo,.mp4,.mov,.avi"
                className="hidden"
                onChange={(event) => {
                  const files = Array.from(event.currentTarget.files ?? [])
                  if (files.length) uploadVideo.mutate(files)
                  event.currentTarget.value = ""
                }}
                multiple
                ref={uploadInput}
                type="file"
              />
              <Input
                accept=".vtt,text/vtt"
                className="hidden"
                onChange={(event) => {
                  const file = event.currentTarget.files?.[0]
                  if (file && selectedVideo) {
                    if (captionObjectUrl.current) {
                      URL.revokeObjectURL(captionObjectUrl.current)
                    }
                    const url = URL.createObjectURL(file)
                    captionObjectUrl.current = url
                    setCaptionTrack({ videoId: selectedVideo.id, url })
                  }
                  event.currentTarget.value = ""
                }}
                ref={captionsInput}
                type="file"
              />
            </div>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            {selectedVideo ? (
              <>
                <div
                  className="relative w-full overflow-hidden rounded-xl bg-black"
                  style={{ aspectRatio }}
                >
                  <video
                    className="absolute inset-0 size-full object-contain"
                    onDurationChange={(event) =>
                      setDuration(event.currentTarget.duration || 0)
                    }
                    onLoadedMetadata={(event) => {
                      const video = event.currentTarget
                      playedMomentIds.current.clear()
                      setDuration(video.duration || 0)
                      if (video.videoWidth && video.videoHeight) {
                        setAspectRatio(video.videoWidth / video.videoHeight)
                      }
                      setCurrentTime(video.currentTime)
                    }}
                    onSeeking={(event) =>
                      handleVideoSeeking(event.currentTarget)
                    }
                    onPause={() => setIsPlaying(false)}
                    onPlay={() => setIsPlaying(true)}
                    onTimeUpdate={(event) =>
                      handleVideoTimeUpdate(event.currentTarget)
                    }
                    onError={() =>
                      setPlaybackError(
                        "This video could not be decoded by the browser. Try an H.264/AAC MP4 file.",
                      )
                    }
                    preload="metadata"
                    ref={videoRef}
                  >
                    <track
                      default={captionTrack?.videoId === selectedVideo.id}
                      kind="captions"
                      label="Captions"
                      src={
                        captionTrack?.videoId === selectedVideo.id
                          ? captionTrack.url
                          : undefined
                      }
                      srcLang="en"
                    />
                  </video>
                  <svg
                    aria-label="Video annotation canvas"
                    className={`absolute inset-0 size-full touch-none ${
                      tool === "select" ? "cursor-default" : "cursor-crosshair"
                    }`}
                    onDoubleClick={onTextDoubleClick}
                    onPointerCancel={onDrawingEnd}
                    onPointerDown={onDrawingStart}
                    onPointerMove={onDrawingMove}
                    onPointerUp={onDrawingEnd}
                    preserveAspectRatio="none"
                    role="img"
                    viewBox={`0 0 1000 ${viewBoxHeight}`}
                  >
                    <defs>
                      {projectedDrawings
                        .filter(
                          (annotation) =>
                            annotation.type === "rectangle" &&
                            [
                              "hatch",
                              "hatch-horizontal",
                              "hatch-cross",
                            ].includes(annotation.rectangleVariant ?? "box"),
                        )
                        .map((annotation) => (
                          <pattern
                            height="12"
                            id={`telestrator-pattern-${annotation.id}`}
                            key={`pattern-${annotation.id}`}
                            patternUnits="userSpaceOnUse"
                            width="12"
                          >
                            <path
                              d={
                                annotation.type === "rectangle" &&
                                annotation.rectangleVariant ===
                                  "hatch-horizontal"
                                  ? "M 0 3 H 12 M 0 9 H 12"
                                  : annotation.type === "rectangle" &&
                                      annotation.rectangleVariant ===
                                        "hatch-cross"
                                    ? "M -3 3 L 3 -3 M 0 12 L 12 0 M 9 15 L 15 9 M 0 0 L 12 12 M -3 9 L 3 15 M 9 -3 L 15 3"
                                    : "M -3 3 L 3 -3 M 0 12 L 12 0 M 9 15 L 15 9"
                              }
                              fill="none"
                              stroke={annotation.color}
                              strokeOpacity="0.52"
                              strokeWidth="2"
                            />
                          </pattern>
                        ))}
                      {projectedDrawings
                        .filter((annotation) => annotation.type === "arrow")
                        .map((annotation) => (
                          <marker
                            id={`telestrator-arrow-${annotation.id}`}
                            key={annotation.id}
                            markerHeight="8"
                            markerWidth="8"
                            orient="auto"
                            refX="7"
                            refY="4"
                            viewBox="0 0 8 8"
                          >
                            <path
                              d="M 0 0 L 8 4 L 0 8 z"
                              fill={annotation.color}
                            />
                          </marker>
                        ))}
                    </defs>
                    {projectedDrawings.map(renderAnnotation)}
                    {visibleAnnotations.map(renderEditHandles)}
                  </svg>
                </div>

                <div className="flex flex-wrap items-center gap-2">
                  <Button
                    aria-label="Go back one frame"
                    onClick={() => seek(currentTime - 1 / 25)}
                    size="icon"
                    variant="outline"
                  >
                    <SkipBack className="size-4" />
                  </Button>
                  <Button
                    aria-label={isPlaying ? "Pause video" : "Play video"}
                    onClick={togglePlayback}
                    size="icon"
                  >
                    {isPlaying ? (
                      <Pause className="size-4" />
                    ) : (
                      <Play className="size-4" />
                    )}
                  </Button>
                  <Button
                    aria-label="Go forward one frame"
                    onClick={() => seek(currentTime + 1 / 25)}
                    size="icon"
                    variant="outline"
                  >
                    <SkipForward className="size-4" />
                  </Button>
                  <Button
                    aria-pressed={Boolean(editingMomentId)}
                    onClick={() =>
                      editingMomentId
                        ? finishFreezeMoment(true)
                        : startFreezeMoment()
                    }
                    size="sm"
                    variant="outline"
                  >
                    <Pause className="size-4" />
                    {editingMomentId ? "Finish freeze" : "Freeze frame"}
                  </Button>
                  <Button
                    aria-label={muted ? "Unmute video" : "Mute video"}
                    onClick={() => {
                      const video = videoRef.current
                      if (!video) return
                      const nextMuted = !muted
                      setMuted(nextMuted)
                      video.muted = nextMuted
                    }}
                    size="icon"
                    variant="outline"
                  >
                    {muted ? "🔇" : "🔊"}
                  </Button>
                  <input
                    aria-label="Video volume"
                    className="w-20 accent-primary"
                    max={1}
                    min={0}
                    onChange={(event) => {
                      const nextVolume = Number(event.currentTarget.value)
                      setVolume(nextVolume)
                      if (videoRef.current) videoRef.current.volume = nextVolume
                      if (nextVolume > 0 && muted) {
                        setMuted(false)
                        if (videoRef.current) videoRef.current.muted = false
                      }
                    }}
                    step={0.05}
                    type="range"
                    value={volume}
                  />
                  <span className="min-w-24 font-mono text-xs tabular-nums">
                    {formatTime(currentTime)} / {formatTime(duration)}
                  </span>
                  <input
                    aria-label="Video timeline"
                    className="h-2 min-w-36 flex-1 cursor-pointer accent-primary"
                    max={duration || 0}
                    min={0}
                    onChange={(event) =>
                      seek(Number(event.currentTarget.value))
                    }
                    step={0.01}
                    type="range"
                    value={currentTime}
                  />
                </div>
                {selectedMoments.length ? (
                  <fieldset
                    aria-label="Freeze moments timeline"
                    className="relative h-5 touch-none"
                    onPointerCancel={finishMomentDrag}
                    onPointerDown={(event) => {
                      const target = event.target
                      const button =
                        target instanceof HTMLElement
                          ? target.closest<HTMLElement>("[data-freeze-moment]")
                          : undefined
                      const momentId = button?.dataset.freezeMoment
                      if (!momentId) return
                      event.preventDefault()
                      event.currentTarget.setPointerCapture(event.pointerId)
                      momentDrag.current = {
                        id: momentId,
                        pointerId: event.pointerId,
                        moved: false,
                      }
                    }}
                    onPointerMove={(event) => {
                      const drag = momentDrag.current
                      if (!drag || drag.pointerId !== event.pointerId) return
                      const bounds = event.currentTarget.getBoundingClientRect()
                      const ratio = Math.min(
                        1,
                        Math.max(
                          0,
                          (event.clientX - bounds.left) / bounds.width,
                        ),
                      )
                      const time = Math.round(ratio * duration * 100) / 100
                      const moment = selectedMoments.find(
                        (item) => item.id === drag.id,
                      )
                      if (!moment || Math.abs(moment.time - time) < 0.02) return
                      drag.moved = true
                      setMoments((current) =>
                        current.map((item) =>
                          item.id === drag.id ? { ...item, time } : item,
                        ),
                      )
                      setAnnotations((current) =>
                        current.map((annotation) =>
                          annotation.eventId === drag.id
                            ? { ...annotation, time }
                            : annotation,
                        ),
                      )
                    }}
                    onPointerUp={finishMomentDrag}
                  >
                    <span className="absolute inset-x-0 top-2 h-px bg-border" />
                    {selectedMoments.map((moment) => (
                      <button
                        aria-label={`Freeze moment at ${formatTime(moment.time)}`}
                        className={`absolute top-0 h-4 w-3 -translate-x-1/2 rounded-sm border border-background ${
                          moment.id === editingMomentId ||
                          moment.id === playbackMomentId
                            ? "bg-amber-400"
                            : "bg-primary"
                        }`}
                        data-freeze-moment={moment.id}
                        key={moment.id}
                        style={{
                          left: `${duration ? (moment.time / duration) * 100 : 0}%`,
                        }}
                        title={`${formatTime(moment.time)} · ${moment.duration}s`}
                        type="button"
                      />
                    ))}
                  </fieldset>
                ) : null}
                {editingMomentId ? (
                  <div className="flex flex-wrap items-center gap-3 rounded-lg border border-amber-500/40 bg-amber-500/10 p-3">
                    <span className="text-sm font-medium">
                      Frozen moment ·{" "}
                      {formatTime(
                        moments.find((moment) => moment.id === editingMomentId)
                          ?.time ?? currentTime,
                      )}
                    </span>
                    <Label className="flex items-center gap-2 text-xs">
                      Show drawings
                      <input
                        aria-label="Freeze moment display duration in seconds"
                        className="w-24 accent-primary"
                        max={30}
                        min={1}
                        onChange={(event) =>
                          updateFreezeDuration(
                            Number(event.currentTarget.value),
                          )
                        }
                        type="range"
                        value={
                          moments.find(
                            (moment) => moment.id === editingMomentId,
                          )?.duration ?? annotationDuration
                        }
                      />
                      <span>{annotationDuration}s</span>
                    </Label>
                    <Button
                      onClick={deleteFreezeMoment}
                      size="sm"
                      variant="outline"
                    >
                      Delete moment
                    </Button>
                    <Button onClick={() => finishFreezeMoment(true)} size="sm">
                      Done · resume
                    </Button>
                  </div>
                ) : null}
                {playbackError ? (
                  <Alert variant="destructive">
                    <AlertTitle>Video playback unavailable</AlertTitle>
                    <AlertDescription>{playbackError}</AlertDescription>
                  </Alert>
                ) : null}
              </>
            ) : (
              <div className="flex min-h-72 flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-border bg-muted/20 p-8 text-center">
                <Video className="size-10 text-muted-foreground" />
                <div>
                  <p className="font-medium">Upload match footage to begin</p>
                  <p className="mt-1 text-sm text-muted-foreground">
                    Your video will be stored with this project.
                  </p>
                </div>
              </div>
            )}

            <div className="flex flex-wrap items-center gap-2 border-t border-border/70 pt-4">
              {(
                [
                  ["select", "Select", ArrowRight],
                  ["freehand", "Freehand", Pencil],
                  ["line", "Line", Pencil],
                  ["arrow", "Arrow", ArrowRight],
                  ["circle", "Ring", Circle],
                  ["rectangle", "Rectangle", RectangleHorizontal],
                  ["funnel", "Funnel / spotlight", Triangle],
                  ["text", "Text", Type],
                ] as const
              ).map(([mode, label, Icon]) => (
                <Button
                  aria-pressed={tool === mode}
                  key={mode}
                  onClick={() => setTool(mode)}
                  size="sm"
                  variant={tool === mode ? "default" : "outline"}
                >
                  <Icon className="size-4" />
                  {label}
                </Button>
              ))}
              <div className="flex flex-wrap items-center gap-3">
                {tool === "arrow" ? (
                  <>
                    <Label className="flex items-center gap-2 text-xs">
                      Arrow style
                      <select
                        aria-label="Arrow style"
                        className="h-9 rounded-md border border-input bg-background px-2 text-sm"
                        onChange={(event) =>
                          setArrowVariant(
                            event.currentTarget.value as ArrowVariant,
                          )
                        }
                        value={arrowVariant}
                      >
                        <option value="solid">Solid</option>
                        <option value="curve">Curved</option>
                        <option value="lob">Lob / high pass</option>
                        <option value="marker">Marker</option>
                        <option value="bounce">Animated bounce</option>
                      </select>
                    </Label>
                    <Label className="flex items-center gap-1 text-xs">
                      <input
                        checked={dashedArrow}
                        onChange={(event) =>
                          setDashedArrow(event.currentTarget.checked)
                        }
                        type="checkbox"
                      />
                      Dashed
                    </Label>
                  </>
                ) : null}
                {tool === "circle" ? (
                  <Label className="flex items-center gap-2 text-xs">
                    Ring style
                    <select
                      aria-label="Ring style"
                      className="h-9 rounded-md border border-input bg-background px-2 text-sm"
                      onChange={(event) =>
                        setRingVariant(event.currentTarget.value as RingVariant)
                      }
                      value={ringVariant}
                    >
                      <option value="ground">Ground ring</option>
                      <option value="dashed">Dashed</option>
                      <option value="radar">Radar</option>
                      <option value="radar-animated">Animated radar</option>
                      <option value="glow">Glow</option>
                      <option value="glow-pulse">Pulsing glow</option>
                      <option value="shockwave">Shockwave</option>
                      <option value="pulse">Pulse</option>
                      <option value="team">Team / linked players</option>
                      <option value="spotlight">Spotlight ring</option>
                    </select>
                  </Label>
                ) : null}
                {tool === "rectangle" ? (
                  <Label className="flex items-center gap-2 text-xs">
                    Area style
                    <select
                      aria-label="Area style"
                      className="h-9 rounded-md border border-input bg-background px-2 text-sm"
                      onChange={(event) =>
                        setRectangleVariant(
                          event.currentTarget.value as RectangleVariant,
                        )
                      }
                      value={rectangleVariant}
                    >
                      <option value="box">Rectangle</option>
                      <option value="dash">Dashed</option>
                      <option value="hatch">Diagonal hatch</option>
                      <option value="hatch-horizontal">Horizontal hatch</option>
                      <option value="hatch-cross">Cross hatch</option>
                      <option value="square">Square</option>
                    </select>
                  </Label>
                ) : null}
                {tool === "text" ? (
                  <Label className="flex items-center gap-2 text-xs">
                    Text style
                    <select
                      aria-label="Text style"
                      className="h-9 rounded-md border border-input bg-background px-2 text-sm"
                      onChange={(event) =>
                        setTextVariant(event.currentTarget.value as TextVariant)
                      }
                      value={textVariant}
                    >
                      <option value="label">Label background</option>
                      <option value="plain">Plain text</option>
                    </select>
                  </Label>
                ) : null}
                {tool === "line" ? (
                  <Label className="flex items-center gap-1 text-xs">
                    <input
                      checked={dashedArrow}
                      onChange={(event) =>
                        setDashedArrow(event.currentTarget.checked)
                      }
                      type="checkbox"
                    />
                    Dashed line
                  </Label>
                ) : null}
                <Label className="flex items-center gap-2 text-xs">
                  Color
                  <input
                    aria-label="Annotation color"
                    className="size-8 cursor-pointer rounded border-0 bg-transparent p-0"
                    onChange={(event) => setColor(event.currentTarget.value)}
                    type="color"
                    value={color}
                  />
                </Label>
                <Label className="flex items-center gap-2 text-xs">
                  Show
                  <input
                    aria-label="Annotation display duration in seconds"
                    className="w-20 accent-primary"
                    max={30}
                    min={1}
                    onChange={(event) =>
                      setAnnotationDuration(Number(event.currentTarget.value))
                    }
                    type="range"
                    value={annotationDuration}
                  />
                  <span className="w-8 tabular-nums">
                    {annotationDuration}s
                  </span>
                </Label>
                <Label className="flex items-center gap-2 text-xs">
                  Size
                  <input
                    aria-label="Annotation line size"
                    className="w-20 accent-primary"
                    max={12}
                    min={2}
                    onChange={(event) =>
                      setLineSize(Number(event.currentTarget.value))
                    }
                    type="range"
                    value={lineSize}
                  />
                </Label>
              </div>
            </div>
            <div className="flex flex-wrap gap-2">
              <Button
                disabled={!undoHistory.current.length}
                onClick={undo}
                size="sm"
                variant="outline"
              >
                <Undo2 className="size-4" />
                Undo
              </Button>
              <Button
                disabled={!selectedAnnotationId}
                onClick={removeSelected}
                size="sm"
                variant="outline"
              >
                <Trash2 className="size-4" />
                Delete selected
              </Button>
              <Button
                disabled={
                  !selectedVideo ||
                  !annotations.some((item) => item.videoId === selectedVideo.id)
                }
                onClick={clearAnnotations}
                size="sm"
                variant="outline"
              >
                <Trash2 className="size-4" />
                Clear video annotations
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card className="border-border/80">
          <CardHeader>
            <CardTitle>Videos and moments</CardTitle>
            <CardDescription>
              Select a clip or jump to a saved annotation timestamp.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            {videos.length ? (
              <div className="flex max-h-52 flex-col gap-2 overflow-y-auto">
                {videos.map((video) => (
                  <Button
                    className="h-auto justify-start whitespace-normal text-left"
                    key={video.id}
                    onClick={() => {
                      setSelectedVideoId(video.id)
                      setCurrentTime(0)
                      setSelectedAnnotationId("")
                    }}
                    variant={
                      selectedVideo?.id === video.id ? "secondary" : "outline"
                    }
                  >
                    <Video className="size-4 shrink-0" />
                    <span className="min-w-0 truncate">{video.filename}</span>
                  </Button>
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">
                No videos have been uploaded to this project yet.
              </p>
            )}
            <div className="border-t border-border/70 pt-4">
              <h3 className="mb-3 text-sm font-semibold">Freeze moments</h3>
              {selectedMoments.length ? (
                <div className="flex max-h-48 flex-col gap-1 overflow-y-auto">
                  {selectedMoments.map((moment) => (
                    <Button
                      className="h-auto justify-start whitespace-normal text-left"
                      key={moment.id}
                      onClick={() => openMoment(moment)}
                      size="sm"
                      variant={
                        moment.id === editingMomentId ? "secondary" : "ghost"
                      }
                    >
                      <span className="size-2.5 shrink-0 rounded-full bg-amber-400" />
                      <span className="font-mono text-xs">
                        {formatTime(moment.time)}
                      </span>
                      <span>{moment.duration}s freeze</span>
                    </Button>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">
                  Freeze a frame to add the first saved moment.
                </p>
              )}
            </div>
            <div className="border-t border-border/70 pt-4">
              <h3 className="mb-3 text-sm font-semibold">Saved annotations</h3>
              {annotations.filter(
                (annotation) => annotation.videoId === selectedVideo?.id,
              ).length ? (
                <div className="flex max-h-80 flex-col gap-2 overflow-y-auto">
                  {annotations
                    .filter(
                      (annotation) => annotation.videoId === selectedVideo?.id,
                    )
                    .map((annotation) => (
                      <Button
                        className="h-auto justify-start whitespace-normal text-left"
                        key={annotation.id}
                        onClick={() => {
                          const moment = moments.find(
                            (item) => item.id === annotation.eventId,
                          )
                          if (moment) openMoment(moment)
                          else seek(annotation.time)
                        }}
                        size="sm"
                        variant="ghost"
                      >
                        <span
                          aria-hidden="true"
                          className="size-2.5 shrink-0 rounded-full"
                          style={{ backgroundColor: annotation.color }}
                        />
                        <span className="font-mono text-xs">
                          {formatTime(annotation.time)}
                        </span>
                        <span className="capitalize">
                          {annotation.type}
                          {annotation.type === "text"
                            ? `: ${annotation.text}`
                            : ""}
                        </span>
                      </Button>
                    ))}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">
                  Draw on a paused frame to add the first moment.
                </p>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
