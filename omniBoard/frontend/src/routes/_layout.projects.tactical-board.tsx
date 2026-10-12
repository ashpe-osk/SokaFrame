import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { createFileRoute, Link, useNavigate } from "@tanstack/react-router"
import { ArrowRight, ClipboardList, LoaderCircle, Plus } from "lucide-react"
import { useState } from "react"
import { client } from "@/client/client.gen"
import { ProjectCategories } from "@/components/Projects/ProjectCategories"
import {
  createDefaultBoard,
  TacticalBoardEditor,
} from "@/components/TacticalBoard/TacticalBoardEditor"
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

type ClientProjectType = "board" | "media"
type ClientProject = {
  id: string
  type: ClientProjectType
  kind: string
  workspace?: string
  title: string
  createdAt: string
  updatedAt: string
}
type ClientProjectList = { projects: ClientProject[] }

export const Route = createFileRoute("/_layout/projects/tactical-board")({
  component: TacticalBoardProjectsPage,
  validateSearch: (search: Record<string, unknown>) => ({
    projectId:
      typeof search.projectId === "string" ? search.projectId : undefined,
    projectType: search.projectType === "media" ? "media" : "board",
  }),
  head: () => ({
    meta: [{ title: "Tactical Board Projects | SokaFrame" }],
  }),
})

function projectPayload(
  id: string,
  title: string,
  updatedAt: string,
): Record<string, unknown> {
  return {
    schemaVersion: 1,
    id,
    type: "media",
    kind: "tactical",
    title,
    createdAt: updatedAt,
    updatedAt,
    settings: {},
    boardState: {
      tokens: [],
      reactEditor: createDefaultBoard(),
    },
    mediaState: null,
    analysisState: null,
  }
}

function formatDate(value: string) {
  const date = new Date(value)
  return Number.isNaN(date.getTime())
    ? "Recently updated"
    : `Updated ${date.toLocaleDateString()}`
}

function TacticalBoardProjectsPage() {
  const { projectId, projectType } = Route.useSearch()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [projectName, setProjectName] = useState("")
  const projectsQuery = useQuery({
    queryKey: ["tactical-projects"],
    queryFn: async () => {
      const [board, media] = await Promise.all(
        (["board", "media"] as const).map(async (type) => {
          const response = await client.get<{ "200": ClientProjectList }>({
            url: `/api/projects/${type}`,
            responseType: "json",
            security: [{ scheme: "bearer", type: "http" }],
          })
          if (!response.data) {
            throw new Error("The server returned no tactical project list.")
          }
          return response.data.projects
        }),
      )
      return [...board, ...media]
        .filter(
          (project) =>
            project.kind === "tactical" && project.workspace !== "telestrator",
        )
        .sort(
          (left, right) =>
            new Date(right.updatedAt).getTime() -
            new Date(left.updatedAt).getTime(),
        )
    },
  })

  const createProject = useMutation({
    mutationFn: async (title: string) => {
      const id = crypto.randomUUID()
      const updatedAt = new Date().toISOString()
      await client.put({
        url: `/api/projects/media/${encodeURIComponent(id)}`,
        body: { payload: projectPayload(id, title, updatedAt) },
        responseType: "json",
        security: [{ scheme: "bearer", type: "http" }],
      })
      return {
        id,
        type: "media" as const,
        kind: "tactical",
        title,
        createdAt: updatedAt,
        updatedAt,
      }
    },
    onSuccess: async (project) => {
      setProjectName("")
      await queryClient.invalidateQueries({ queryKey: ["tactical-projects"] })
      await navigate({
        to: "/projects/tactical-board",
        search: { projectId: project.id, projectType: project.type },
      })
    },
  })

  if (projectId) {
    const selectedProject = projectsQuery.data?.find(
      (project) => project.id === projectId && project.type === projectType,
    )
    if (projectsQuery.isPending) {
      return (
        <p
          className="flex items-center gap-2 text-sm text-muted-foreground"
          role="status"
        >
          <LoaderCircle className="size-4 animate-spin" />
          Loading tactical projects…
        </p>
      )
    }
    if (projectsQuery.isError) {
      return (
        <Alert variant="destructive">
          <AlertTitle>Projects unavailable</AlertTitle>
          <AlertDescription>{projectsQuery.error.message}</AlertDescription>
        </Alert>
      )
    }
    if (selectedProject) {
      return (
        <TacticalBoardEditor
          key={`${selectedProject.type}-${selectedProject.id}`}
          projectId={selectedProject.id}
          projectName={selectedProject.title}
          projectType={selectedProject.type}
        />
      )
    }
  }

  return (
    <div className="flex flex-col gap-7">
      <header>
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-primary">
          Projects
        </p>
        <h1 className="mt-1 text-3xl font-bold tracking-tight sm:text-4xl">
          Tactical Board
        </h1>
        <p className="mt-2 max-w-2xl text-sm text-muted-foreground sm:text-base">
          Create a board workspace to plan formations, player positions, and
          movement.
        </p>
      </header>

      <ProjectCategories active="tactical-board" />

      {projectId && !projectsQuery.isPending && !projectsQuery.isError ? (
        <Alert variant="destructive">
          <AlertTitle>Tactical project unavailable</AlertTitle>
          <AlertDescription>
            This project could not be found. Choose another board or create a
            new one below.
          </AlertDescription>
        </Alert>
      ) : null}

      <Card className="border-border/80">
        <CardHeader>
          <CardTitle>Create a tactical project</CardTitle>
          <CardDescription>
            Give your board a name, then open it to start planning.
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
              <Label htmlFor="tactical-project-name">Project name</Label>
              <Input
                id="tactical-project-name"
                maxLength={120}
                onChange={(event) => setProjectName(event.target.value)}
                placeholder="e.g. Matchday 12 — pressing plan"
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
              <AlertDescription>{createProject.error.message}</AlertDescription>
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
          Loading tactical projects…
        </p>
      ) : projectsQuery.isError ? (
        <Alert variant="destructive">
          <AlertTitle>Projects unavailable</AlertTitle>
          <AlertDescription>{projectsQuery.error.message}</AlertDescription>
        </Alert>
      ) : projectsQuery.data.length ? (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {projectsQuery.data.map((project) => (
            <Card
              className="group border-border/80 transition-all hover:-translate-y-0.5 hover:border-primary/40 hover:shadow-lg hover:shadow-black/10"
              key={`${project.type}-${project.id}`}
            >
              <CardHeader className="flex-row items-start gap-3">
                <span className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-primary/12 text-primary">
                  <ClipboardList className="size-5" aria-hidden="true" />
                </span>
                <div className="min-w-0">
                  <CardTitle className="truncate text-base">
                    {project.title}
                  </CardTitle>
                  <CardDescription className="mt-1">
                    {formatDate(project.updatedAt)}
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
                    search={{
                      projectId: project.id,
                      projectType: project.type,
                    }}
                    to="/projects/tactical-board"
                  >
                    Open board
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
            <CardTitle>No tactical projects yet</CardTitle>
            <CardDescription>
              Create your first tactical project to start building a match plan.
            </CardDescription>
          </CardHeader>
        </Card>
      )}
    </div>
  )
}
