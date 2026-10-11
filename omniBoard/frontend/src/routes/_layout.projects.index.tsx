import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { createFileRoute, Link, useNavigate } from "@tanstack/react-router"
import { ArrowUpRight, CalendarDays, FolderKanban, Plus } from "lucide-react"
import { useState } from "react"

import { ProjectsService } from "@/client"
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
import useCustomToast from "@/hooks/useCustomToast"
import { handleError } from "@/utils"

export const Route = createFileRoute("/_layout/projects/")({
  component: ProjectsPage,
  head: () => ({ meta: [{ title: "Projects | SokaFrame" }] }),
})

function ProjectsPage() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const { showErrorToast, showSuccessToast } = useCustomToast()
  const [name, setName] = useState("")
  const [description, setDescription] = useState("")
  const [formError, setFormError] = useState("")
  const projectsQuery = useQuery({
    queryKey: ["projects"],
    queryFn: async () =>
      (await ProjectsService.readProjects({ query: { skip: 0, limit: 100 } }))
        .data,
  })
  const createProject = useMutation({
    mutationFn: () =>
      ProjectsService.createProject({
        body: { name: name.trim(), description: description.trim() || null },
      }),
    onSuccess: async ({ data: project }) => {
      showSuccessToast("Project created")
      await queryClient.invalidateQueries({ queryKey: ["projects"] })
      await navigate({
        to: "/projects/$projectId",
        params: { projectId: project.id },
      })
    },
    onError: handleError.bind(showErrorToast),
  })

  const submitProject = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const trimmedName = name.trim()
    if (!trimmedName) {
      setFormError("Enter a project name.")
      return
    }
    if (trimmedName.length > 120) {
      setFormError("Project names must be 120 characters or fewer.")
      return
    }
    setFormError("")
    createProject.mutate()
  }

  return (
    <div className="flex flex-col gap-8">
      <header className="flex flex-col gap-2">
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-primary">
          Workspace
        </p>
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h1 className="text-3xl font-bold tracking-tight sm:text-4xl">
              Projects
            </h1>
            <p className="mt-2 max-w-2xl text-sm text-muted-foreground sm:text-base">
              Organize match footage, video analysis, and team insights in one
              place.
            </p>
          </div>
          {projectsQuery.data && (
            <span className="rounded-full border bg-card px-3 py-1 text-xs font-medium text-muted-foreground">
              {projectsQuery.data.count}{" "}
              {projectsQuery.data.count === 1 ? "project" : "projects"}
            </span>
          )}
        </div>
      </header>

      <Card className="gap-5 border-border/80 bg-gradient-to-br from-card to-card/70 shadow-md shadow-black/10">
        <CardHeader className="gap-2">
          <div className="flex items-center gap-3">
            <span className="flex size-10 items-center justify-center rounded-xl bg-primary/12 text-primary">
              <Plus className="size-5" aria-hidden="true" />
            </span>
            <div className="grid gap-1">
              <CardTitle className="text-lg">Create a project</CardTitle>
              <CardDescription>
                Start a workspace for a match or video collection.
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <form
            className="grid items-end gap-4 lg:grid-cols-[minmax(0,0.85fr)_minmax(0,1.4fr)_auto]"
            onSubmit={submitProject}
          >
            <div className="grid gap-2">
              <Label className="text-sm font-medium" htmlFor="project-name">
                Project name
              </Label>
              <Input
                className="h-11 border-input/80 bg-background/70 placeholder:text-muted-foreground/70 focus-visible:border-primary"
                id="project-name"
                maxLength={120}
                onChange={(event) => setName(event.target.value)}
                placeholder="Season match analysis"
                required
                value={name}
              />
            </div>
            <div className="grid gap-2">
              <Label
                className="text-sm font-medium"
                htmlFor="project-description"
              >
                Description (optional)
              </Label>
              <Input
                className="h-11 border-input/80 bg-background/70 placeholder:text-muted-foreground/70 focus-visible:border-primary"
                id="project-description"
                maxLength={2000}
                onChange={(event) => setDescription(event.target.value)}
                placeholder="Add a short note"
                value={description}
              />
            </div>
            <Button
              className="h-11 w-full px-5 shadow-sm shadow-primary/15 transition-colors lg:w-auto"
              disabled={createProject.isPending}
            >
              <Plus />
              {createProject.isPending ? "Creating…" : "Create project"}
            </Button>
          </form>
          {formError && (
            <p className="mt-3 text-sm text-destructive" role="alert">
              {formError}
            </p>
          )}
        </CardContent>
      </Card>

      {projectsQuery.isPending ? (
        <p role="status">Loading projects…</p>
      ) : projectsQuery.isError ? (
        <p className="text-destructive" role="alert">
          Projects could not be loaded. {projectsQuery.error.message}
        </p>
      ) : projectsQuery.data.data.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-border/80 bg-card/40 px-6 py-12 text-center sm:py-16">
          <span className="mx-auto mb-4 flex size-12 items-center justify-center rounded-2xl bg-primary/10 text-primary">
            <FolderKanban className="size-6" aria-hidden="true" />
          </span>
          <h2 className="text-lg font-semibold">No projects yet</h2>
          <p className="mx-auto mt-1 max-w-sm text-sm text-muted-foreground">
            Create a project above to upload your first video.
          </p>
        </div>
      ) : (
        <section
          aria-label="Your projects"
          className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3"
        >
          {projectsQuery.data.data.map((project) => (
            <Link
              className="group rounded-2xl focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 focus-visible:ring-offset-background"
              key={project.id}
              params={{ projectId: project.id }}
              to="/projects/$projectId"
            >
              <Card className="h-full gap-5 border-border/80 bg-card/80 shadow-sm transition-all duration-200 group-hover:-translate-y-0.5 group-hover:border-primary/50 group-hover:bg-card group-hover:shadow-lg group-hover:shadow-black/15">
                <CardHeader className="flex flex-row items-start justify-between gap-4">
                  <span className="flex size-11 shrink-0 items-center justify-center rounded-xl border border-primary/15 bg-primary/10 text-primary">
                    <FolderKanban className="size-5" aria-hidden="true" />
                  </span>
                  <ArrowUpRight
                    className="mt-1 size-4 text-muted-foreground transition-colors group-hover:text-primary"
                    aria-hidden="true"
                  />
                </CardHeader>
                <CardContent className="grid gap-4">
                  <div className="grid gap-2">
                    <CardTitle className="line-clamp-1 text-base transition-colors group-hover:text-primary">
                      {project.name}
                    </CardTitle>
                    <CardDescription className="line-clamp-2 min-h-10 leading-relaxed">
                      {project.description || "No description"}
                    </CardDescription>
                  </div>
                  <div className="flex items-center gap-2 border-t border-border/70 pt-3 text-xs text-muted-foreground">
                    <CalendarDays className="size-3.5" aria-hidden="true" />
                    <span>
                      Updated{" "}
                      {new Date(project.updated_at).toLocaleDateString()}
                    </span>
                  </div>
                </CardContent>
              </Card>
            </Link>
          ))}
        </section>
      )}
    </div>
  )
}
