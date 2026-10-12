import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Link } from "@tanstack/react-router"
import {
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  ArrowUp,
  Brush,
  Circle,
  Download,
  Eraser,
  Plus,
  RectangleHorizontal,
  Save,
  TextCursorInput,
  Trash2,
} from "lucide-react"
import { useEffect, useMemo, useRef, useState } from "react"

import { client } from "@/client/client.gen"
import { ProjectCategories } from "@/components/Projects/ProjectCategories"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

type TeamSide = "home" | "away"
type ToolMode =
  | "select"
  | "arrow"
  | "freehand"
  | "circle"
  | "rectangle"
  | "text"
  | "eraser"

type BoardPoint = { x: number; y: number }
type BoardDrawing =
  | {
      id: string
      type: "freehand"
      color: string
      size?: number
      points: BoardPoint[]
    }
  | {
      id: string
      type: "circle" | "rectangle"
      color: string
      size?: number
      start: BoardPoint
      end: BoardPoint
    }
  | {
      id: string
      type: "text"
      color: string
      size?: number
      point: BoardPoint
      text: string
    }

type Player = {
  id: string
  name: string
  number: number
  x: number
  y: number
}

type TacticalArrow = {
  id: string
  start: { x: number; y: number }
  end: { x: number; y: number }
}

type BoardState = {
  version: 1
  formations: Record<TeamSide, string>
  home: Player[]
  away: Player[]
  arrows: TacticalArrow[]
  drawings: BoardDrawing[]
}

type ClientProjectType = "board" | "media"
type StoredClientProject = {
  payload: Record<string, unknown>
  kind: string
}

const formations = ["4-3-3", "4-4-2", "3-5-2", "4-2-3-1"] as const
const pitchWidth = 1000
const pitchHeight = 650

function getFormationRows(formation: string): number[] {
  const rows = formation
    .split("-")
    .map(Number)
    .filter((count) => Number.isInteger(count) && count > 0)
  return rows.length ? rows : [4, 3, 3]
}

function createTeam(side: TeamSide, formation: string): Player[] {
  const rows = getFormationRows(formation)
  const xPositions = [side === "home" ? 8 : 92]
  rows.forEach((_, index) => {
    const x = 24 + index * (54 / Math.max(rows.length - 1, 1))
    xPositions.push(side === "home" ? x : 100 - x)
  })

  const players: Player[] = [
    {
      id: `${side}-1`,
      name: "Goalkeeper",
      number: 1,
      x: xPositions[0],
      y: 50,
    },
  ]
  let number = 2

  rows.forEach((count, rowIndex) => {
    for (let index = 0; index < count; index += 1) {
      players.push({
        id: `${side}-${number}`,
        name: `Player ${number}`,
        number,
        x: xPositions[rowIndex + 1],
        y: ((index + 1) / (count + 1)) * 100,
      })
      number += 1
    }
  })

  return players
}

export function createDefaultBoard(): BoardState {
  return {
    version: 1,
    formations: { home: "4-3-3", away: "4-3-3" },
    home: createTeam("home", "4-3-3"),
    away: createTeam("away", "4-3-3"),
    arrows: [],
    drawings: [],
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value)
}

function isPlayer(value: unknown): value is Player {
  if (!isRecord(value)) return false
  const player = value
  return (
    typeof player.id === "string" &&
    typeof player.name === "string" &&
    typeof player.number === "number" &&
    typeof player.x === "number" &&
    typeof player.y === "number" &&
    player.x >= 0 &&
    player.x <= 100 &&
    player.y >= 0 &&
    player.y <= 100
  )
}

function isTacticalArrow(value: unknown): value is TacticalArrow {
  if (!isRecord(value)) return false
  const arrow = value
  if (typeof arrow.id !== "string") return false
  const isPoint = (point: unknown): point is { x: number; y: number } => {
    if (!isRecord(point)) return false
    const coordinates = point
    return (
      typeof coordinates.x === "number" &&
      typeof coordinates.y === "number" &&
      coordinates.x >= 0 &&
      coordinates.x <= 100 &&
      coordinates.y >= 0 &&
      coordinates.y <= 100
    )
  }
  return isPoint(arrow.start) && isPoint(arrow.end)
}

function isBoardPoint(value: unknown): value is BoardPoint {
  return (
    isRecord(value) &&
    typeof value.x === "number" &&
    value.x >= 0 &&
    value.x <= 100 &&
    typeof value.y === "number" &&
    value.y >= 0 &&
    value.y <= 100
  )
}

function isBoardDrawing(value: unknown): value is BoardDrawing {
  if (!isRecord(value) || typeof value.id !== "string") return false
  if (typeof value.color !== "string" || !/^#[\da-f]{6}$/i.test(value.color)) {
    return false
  }
  if (
    value.size !== undefined &&
    (typeof value.size !== "number" || value.size < 7 || value.size > 100)
  ) {
    return false
  }
  if (value.type === "freehand") {
    return (
      Array.isArray(value.points) &&
      value.points.length > 0 &&
      value.points.every(isBoardPoint)
    )
  }
  if (value.type === "text") {
    return (
      typeof value.text === "string" &&
      value.text.length > 0 &&
      value.text.length <= 100 &&
      isBoardPoint(value.point)
    )
  }
  return (
    (value.type === "circle" || value.type === "rectangle") &&
    isBoardPoint(value.start) &&
    isBoardPoint(value.end)
  )
}

function parseBoardState(value: unknown): BoardState {
  if (value === null || value === undefined) return createDefaultBoard()
  if (!isRecord(value)) {
    throw new Error("The saved board does not contain a supported layout.")
  }
  const candidate = value
  if (candidate.reactEditor !== undefined) {
    return parseBoardState(candidate.reactEditor)
  }

  if (Array.isArray(candidate.tokens)) {
    const board: BoardState = {
      version: 1,
      formations: { home: "4-3-3", away: "4-3-3" },
      home: [],
      away: [],
      arrows: [],
      drawings: [],
    }
    if (candidate.drawings !== undefined) {
      if (
        !Array.isArray(candidate.drawings) ||
        !candidate.drawings.every(isBoardDrawing)
      ) {
        throw new Error("The saved board contains unsupported drawings.")
      }
      board.drawings = candidate.drawings
    }
    candidate.tokens.forEach((token, index) => {
      if (
        !isRecord(token) ||
        (token.team !== "home" && token.team !== "away")
      ) {
        return
      }
      const number =
        typeof token.number === "number"
          ? token.number
          : Number.parseInt(String(token.number ?? ""), 10)
      const coordinate = (input: unknown, fallback: number) => {
        const parsed =
          typeof input === "number"
            ? input
            : Number.parseFloat(String(input ?? ""))
        return Number.isFinite(parsed)
          ? Math.max(0, Math.min(100, parsed))
          : fallback
      }
      const playerNumber =
        Number.isInteger(number) && number >= 0 && number <= 99
          ? number
          : index + 1
      const side = token.team
      const player: Player = {
        id: typeof token.id === "string" ? token.id : `${side}-legacy-${index}`,
        name:
          (typeof token.name === "string" && token.name) ||
          (typeof token.label === "string" && token.label) ||
          `Player ${playerNumber}`,
        number: playerNumber,
        x: coordinate(token.x, side === "home" ? 35 : 65),
        y: coordinate(token.y, 50),
      }
      board[side].push(player)
    })
    return board
  }

  const savedFormations = candidate.formations as
    | Record<string, unknown>
    | undefined
  if (
    candidate.version !== 1 ||
    !savedFormations ||
    typeof savedFormations.home !== "string" ||
    typeof savedFormations.away !== "string" ||
    !Array.isArray(candidate.home) ||
    !Array.isArray(candidate.away) ||
    !Array.isArray(candidate.arrows) ||
    !candidate.home.every(isPlayer) ||
    !candidate.away.every(isPlayer) ||
    !candidate.arrows.every(isTacticalArrow)
  ) {
    throw new Error("The saved board uses an unsupported layout format.")
  }
  if (
    !formations.includes(savedFormations.home as (typeof formations)[number]) ||
    !formations.includes(savedFormations.away as (typeof formations)[number])
  ) {
    throw new Error("The saved board contains an unsupported formation.")
  }
  const drawings = candidate.drawings ?? []
  if (!Array.isArray(drawings) || !drawings.every(isBoardDrawing)) {
    throw new Error("The saved board contains unsupported drawings.")
  }
  return {
    version: 1,
    formations: {
      home: savedFormations.home,
      away: savedFormations.away,
    },
    home: candidate.home,
    away: candidate.away,
    arrows: candidate.arrows,
    drawings,
  }
}

function pointFromEvent(event: React.PointerEvent<HTMLElement>) {
  const bounds = event.currentTarget.getBoundingClientRect()
  return {
    x: Math.max(
      3,
      Math.min(97, ((event.clientX - bounds.left) / bounds.width) * 100),
    ),
    y: Math.max(
      3,
      Math.min(97, ((event.clientY - bounds.top) / bounds.height) * 100),
    ),
  }
}

function drawingElement(drawing: BoardDrawing) {
  const strokeWidth =
    drawing.size === undefined
      ? 6
      : Math.max(2, Math.round(drawing.size / 16) + 1)
  if (drawing.type === "freehand") {
    return (
      <polyline
        fill="none"
        key={drawing.id}
        pointerEvents="none"
        points={drawing.points
          .map(
            (point) =>
              `${(point.x / 100) * pitchWidth},${(point.y / 100) * pitchHeight}`,
          )
          .join(" ")}
        stroke={drawing.color}
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={strokeWidth}
      />
    )
  }
  if (drawing.type === "text") {
    return (
      <text
        dominantBaseline="central"
        fill={drawing.color}
        fontFamily="Inter, Arial, sans-serif"
        fontSize={drawing.size === undefined ? 24 : 12 + drawing.size * 0.36}
        fontWeight="700"
        key={drawing.id}
        paintOrder="stroke"
        pointerEvents="none"
        stroke="#11251a"
        strokeLinejoin="round"
        strokeWidth="3"
        x={(drawing.point.x / 100) * pitchWidth}
        y={(drawing.point.y / 100) * pitchHeight}
      >
        {drawing.text}
      </text>
    )
  }
  if (drawing.type === "circle") {
    return (
      <ellipse
        cx={((drawing.start.x + drawing.end.x) / 2 / 100) * pitchWidth}
        cy={((drawing.start.y + drawing.end.y) / 2 / 100) * pitchHeight}
        fill="none"
        key={drawing.id}
        pointerEvents="none"
        rx={(Math.abs(drawing.end.x - drawing.start.x) / 2 / 100) * pitchWidth}
        ry={(Math.abs(drawing.end.y - drawing.start.y) / 2 / 100) * pitchHeight}
        stroke={drawing.color}
        strokeWidth={strokeWidth}
      />
    )
  }
  return (
    <rect
      fill="none"
      height={(Math.abs(drawing.end.y - drawing.start.y) / 100) * pitchHeight}
      key={drawing.id}
      pointerEvents="none"
      stroke={drawing.color}
      strokeWidth={strokeWidth}
      width={(Math.abs(drawing.end.x - drawing.start.x) / 100) * pitchWidth}
      x={(Math.min(drawing.start.x, drawing.end.x) / 100) * pitchWidth}
      y={(Math.min(drawing.start.y, drawing.end.y) / 100) * pitchHeight}
    />
  )
}

function TacticalPitch({
  board,
  side,
  selectedPlayerId,
  mode,
  pendingArrow,
  draftDrawing,
  onPlayerPointerDown,
  onPlayerPointerMove,
  onPlayerPointerUp,
  onPlayerSelect,
  onPitchClick,
  onPitchPointerDown,
  onPitchPointerCancel,
  onPitchKeyDown,
  svgRef,
}: {
  board: BoardState
  side: TeamSide
  selectedPlayerId: string | null
  mode: ToolMode
  pendingArrow: { x: number; y: number } | null
  draftDrawing: BoardDrawing | null
  onPlayerPointerDown: (
    event: React.PointerEvent<HTMLButtonElement>,
    player: Player,
  ) => void
  onPlayerPointerMove: (event: React.PointerEvent<HTMLDivElement>) => void
  onPlayerPointerUp: () => void
  onPlayerSelect: (player: Player, team: TeamSide) => void
  onPitchClick: (event: React.MouseEvent<HTMLDivElement>) => void
  onPitchPointerDown: (event: React.PointerEvent<HTMLDivElement>) => void
  onPitchPointerCancel: () => void
  onPitchKeyDown: (event: React.KeyboardEvent<HTMLDivElement>) => void
  svgRef: React.RefObject<SVGSVGElement | null>
}) {
  const teamColors = {
    home: { fill: "#f47721", text: "#17120e" },
    away: { fill: "#f1f5f9", text: "#142019" },
  }

  return (
    <div
      aria-label="Interactive football pitch. Drag players to move them."
      className={`relative aspect-[20/13] w-full touch-none select-none overflow-hidden rounded-xl ${
        mode === "arrow" || mode === "eraser"
          ? "cursor-crosshair"
          : mode === "freehand" || mode === "circle" || mode === "rectangle"
            ? "cursor-cell"
            : "cursor-default"
      }`}
      onClick={onPitchClick}
      onKeyDown={onPitchKeyDown}
      onPointerDown={onPitchPointerDown}
      onPointerMove={onPlayerPointerMove}
      onPointerUp={onPlayerPointerUp}
      onPointerCancel={onPitchPointerCancel}
      role="application"
    >
      <svg
        aria-hidden="true"
        className="absolute inset-0 size-full"
        ref={svgRef}
        viewBox={`0 0 ${pitchWidth} ${pitchHeight}`}
        xmlns="http://www.w3.org/2000/svg"
      >
        <defs>
          <linearGradient id="pitch-bg" x1="0" x2="1" y1="0" y2="1">
            <stop offset="0" stopColor="#214d36" />
            <stop offset="1" stopColor="#163b2a" />
          </linearGradient>
          <marker
            id="arrow-head"
            markerHeight="7"
            markerWidth="8"
            orient="auto"
            refX="7"
            refY="3.5"
          >
            <path d="M0 0 8 3.5 0 7z" fill="#ffe081" />
          </marker>
        </defs>
        <rect
          fill="url(#pitch-bg)"
          height={pitchHeight}
          rx="20"
          width={pitchWidth}
        />
        {Array.from({ length: 10 }, (_, index) => (
          <rect
            fill={index % 2 === 0 ? "#ffffff08" : "transparent"}
            height={pitchHeight}
            key={index}
            width={pitchWidth / 10}
            x={index * (pitchWidth / 10)}
          />
        ))}
        <g fill="none" opacity=".82" stroke="#e4f2e8" strokeWidth="3">
          <rect height="590" rx="4" width="940" x="30" y="30" />
          <line x1="500" x2="500" y1="30" y2="620" />
          <circle cx="500" cy="325" r="91" />
          <circle cx="500" cy="325" fill="#e4f2e8" r="4" />
          <rect height="250" width="145" x="30" y="200" />
          <rect height="130" width="48" x="30" y="260" />
          <path d="M175 258a92 92 0 0 1 0 134" />
          <circle cx="137" cy="325" fill="#e4f2e8" r="4" />
          <rect height="250" width="145" x="825" y="200" />
          <rect height="130" width="48" x="922" y="260" />
          <path d="M825 258a92 92 0 0 0 0 134" />
          <circle cx="863" cy="325" fill="#e4f2e8" r="4" />
          <path d="M30 55h28v-25M942 30v25h28M30 595h28v25M942 620v-25h28" />
        </g>
        {[...board.drawings, ...(draftDrawing ? [draftDrawing] : [])].map(
          drawingElement,
        )}
        {board.arrows.map((arrow) => (
          <line
            key={arrow.id}
            markerEnd="url(#arrow-head)"
            stroke="#ffe081"
            strokeDasharray="10 7"
            strokeLinecap="round"
            strokeWidth="7"
            x1={(arrow.start.x / 100) * pitchWidth}
            x2={(arrow.end.x / 100) * pitchWidth}
            y1={(arrow.start.y / 100) * pitchHeight}
            y2={(arrow.end.y / 100) * pitchHeight}
          />
        ))}
        {pendingArrow && (
          <circle
            cx={(pendingArrow.x / 100) * pitchWidth}
            cy={(pendingArrow.y / 100) * pitchHeight}
            fill="#ffe081"
            opacity=".8"
            r="9"
          />
        )}
        {(["home", "away"] as const).flatMap((team) =>
          board[team].map((player) => {
            const colors = teamColors[team]
            const x = (player.x / 100) * pitchWidth
            const y = (player.y / 100) * pitchHeight
            const isSelected = selectedPlayerId === player.id && team === side
            return (
              <g
                key={player.id}
                pointerEvents="none"
                transform={`translate(${x} ${y})`}
              >
                {isSelected && (
                  <circle
                    cx="0"
                    cy="0"
                    fill="none"
                    r="31"
                    stroke="#ffe081"
                    strokeDasharray="4 4"
                    strokeWidth="2"
                  />
                )}
                <circle
                  cx="0"
                  cy="0"
                  fill={colors.fill}
                  r="24"
                  stroke={isSelected ? "#fff" : "#101913"}
                  strokeWidth={isSelected ? "4" : "3"}
                />
                <text
                  dominantBaseline="central"
                  fill={colors.text}
                  fontFamily="Inter, Arial, sans-serif"
                  fontSize="18"
                  fontWeight="800"
                  pointerEvents="none"
                  textAnchor="middle"
                  y="1"
                >
                  {player.number}
                </text>
                <text
                  fill="#fff"
                  fontFamily="Inter, Arial, sans-serif"
                  fontSize="14"
                  fontWeight="700"
                  paintOrder="stroke"
                  pointerEvents="none"
                  stroke="#11251a"
                  strokeLinejoin="round"
                  strokeWidth="4"
                  textAnchor="middle"
                  y="43"
                >
                  {player.name.slice(0, 14)}
                </text>
              </g>
            )
          }),
        )}
      </svg>
      {(["home", "away"] as const).flatMap((team) =>
        board[team].map((player) => (
          <button
            aria-label={`${team === "home" ? "Home" : "Away"} ${player.number}, ${player.name}`}
            aria-pressed={selectedPlayerId === player.id && team === side}
            className={`absolute z-10 size-12 -translate-x-1/2 -translate-y-1/2 rounded-full bg-transparent focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white ${
              mode === "select"
                ? "cursor-grab active:cursor-grabbing"
                : "pointer-events-none"
            }`}
            key={`player-control-${player.id}`}
            onClick={(event) => {
              event.stopPropagation()
              onPlayerSelect(player, team)
            }}
            onPointerDown={(event) => onPlayerPointerDown(event, player)}
            style={{ left: `${player.x}%`, top: `${player.y}%` }}
            type="button"
          />
        )),
      )}
    </div>
  )
}

export function TacticalBoardEditor({
  projectId,
  projectName,
  projectType,
}: {
  projectId: string
  projectName: string
  projectType: ClientProjectType
}) {
  const queryClient = useQueryClient()
  const svgRef = useRef<SVGSVGElement>(null)
  const [board, setBoard] = useState<BoardState>(createDefaultBoard)
  const [savedBoard, setSavedBoard] = useState("")
  const [side, setSide] = useState<TeamSide>("home")
  const [mode, setMode] = useState<ToolMode>("select")
  const [drawingColor, setDrawingColor] = useState("#ffe081")
  const [drawingSize, setDrawingSize] = useState(7)
  const [selectedPlayerId, setSelectedPlayerId] = useState<string | null>(null)
  const [draggedPlayer, setDraggedPlayer] = useState<{
    side: TeamSide
    id: string
  } | null>(null)
  const [pendingArrow, setPendingArrow] = useState<{
    x: number
    y: number
  } | null>(null)
  const [draftDrawing, setDraftDrawing] = useState<BoardDrawing | null>(null)

  const boardQuery = useQuery({
    queryKey: ["tactical-board", projectType, projectId],
    queryFn: async () => {
      const response = await client.get<{ "200": StoredClientProject }>({
        url: `/api/projects/${projectType}/${encodeURIComponent(projectId)}`,
        responseType: "json",
        security: [{ scheme: "bearer", type: "http" }],
      })
      const project = response.data
      if (!project) {
        throw new Error("The server returned no tactical project data.")
      }
      if (project.kind !== "tactical") {
        throw new Error("This project is not a tactical board project.")
      }
      return {
        board: parseBoardState(project.payload.boardState),
        payload: project.payload,
      }
    },
  })

  useEffect(() => {
    if (boardQuery.data) {
      setBoard(boardQuery.data.board)
      setSavedBoard(JSON.stringify(boardQuery.data.board))
    }
  }, [boardQuery.data])

  const saveMutation = useMutation({
    mutationFn: async (nextBoard: BoardState) => {
      const currentPayload = boardQuery.data?.payload ?? {}
      const existingBoardState = isRecord(currentPayload.boardState)
        ? currentPayload.boardState
        : {}
      const existingTokens = Array.isArray(existingBoardState.tokens)
        ? existingBoardState.tokens
        : []
      const tokens = (["home", "away"] as const).flatMap((team) =>
        nextBoard[team].map((player) => {
          const prior = existingTokens.find(
            (token) =>
              isRecord(token) &&
              token.team === team &&
              String(token.number) === String(player.number),
          )
          return {
            ...(isRecord(prior) ? prior : {}),
            id: player.id,
            name: player.name,
            number: player.number,
            team,
            x: `${player.x}%`,
            y: `${player.y}%`,
          }
        }),
      )
      const now = new Date().toISOString()
      const payload = {
        ...currentPayload,
        schemaVersion: 1,
        id: projectId,
        type: projectType,
        kind: "tactical",
        title: projectName,
        createdAt:
          typeof currentPayload.createdAt === "string"
            ? currentPayload.createdAt
            : now,
        updatedAt: now,
        settings: isRecord(currentPayload.settings)
          ? currentPayload.settings
          : {},
        boardState: {
          ...existingBoardState,
          tokens,
          reactEditor: nextBoard,
        },
      }
      await client.put({
        url: `/api/projects/${projectType}/${encodeURIComponent(projectId)}`,
        body: { payload },
        responseType: "json",
        security: [{ scheme: "bearer", type: "http" }],
      })
      return nextBoard
    },
    onSuccess: async (_response, nextBoard) => {
      setSavedBoard(JSON.stringify(nextBoard))
      await queryClient.invalidateQueries({
        queryKey: ["tactical-board", projectType, projectId],
        refetchType: "none",
      })
    },
  })

  const isDirty = useMemo(
    () => savedBoard !== JSON.stringify(board),
    [board, savedBoard],
  )
  const selectedPlayer =
    board[side].find((player) => player.id === selectedPlayerId) ?? null

  function updateBoard(update: (current: BoardState) => BoardState) {
    setBoard((current) => update(current))
  }

  function updatePlayer(playerId: string, update: Partial<Player>) {
    updateBoard((current) => ({
      ...current,
      [side]: current[side].map((player) =>
        player.id === playerId ? { ...player, ...update } : player,
      ),
    }))
  }

  function selectTool(nextMode: ToolMode) {
    setMode(nextMode)
    setPendingArrow(null)
    setDraftDrawing(null)
  }

  function updateFormation(formation: string) {
    const positionedPlayers = createTeam(side, formation)
    updateBoard((current) => ({
      ...current,
      formations: { ...current.formations, [side]: formation },
      [side]: [
        ...positionedPlayers.map((positioned, index) => {
          const existing = current[side][index]
          return existing
            ? {
                ...positioned,
                id: existing.id,
                name: existing.name,
                number: existing.number,
              }
            : positioned
        }),
        ...current[side].slice(positionedPlayers.length),
      ],
    }))
    setSelectedPlayerId(null)
  }

  function addPlayer() {
    const players = board[side]
    if (players.length >= 20) return
    const number = Math.max(0, ...players.map((player) => player.number)) + 1
    const player: Player = {
      id: `${side}-${crypto.randomUUID()}`,
      name: `Player ${number}`,
      number,
      x: side === "home" ? 38 : 62,
      y: 50,
    }
    updateBoard((current) => ({
      ...current,
      [side]: [...current[side], player],
    }))
    setSelectedPlayerId(player.id)
  }

  function removeSelectedPlayer() {
    if (!selectedPlayerId) return
    updateBoard((current) => ({
      ...current,
      [side]: current[side].filter((player) => player.id !== selectedPlayerId),
    }))
    setSelectedPlayerId(null)
  }

  function eraseAt(point: BoardPoint) {
    const toPixels = ({ x, y }: BoardPoint) => ({
      x: (x / 100) * pitchWidth,
      y: (y / 100) * pitchHeight,
    })
    const distanceToSegment = (
      pointValue: BoardPoint,
      start: BoardPoint,
      end: BoardPoint,
    ) => {
      const target = toPixels(pointValue)
      const first = toPixels(start)
      const second = toPixels(end)
      const dx = second.x - first.x
      const dy = second.y - first.y
      const lengthSquared = dx * dx + dy * dy
      const ratio =
        lengthSquared === 0
          ? 0
          : Math.max(
              0,
              Math.min(
                1,
                ((target.x - first.x) * dx + (target.y - first.y) * dy) /
                  lengthSquared,
              ),
            )
      return Math.hypot(
        target.x - (first.x + ratio * dx),
        target.y - (first.y + ratio * dy),
      )
    }
    const drawingDistance = (drawing: BoardDrawing) => {
      if (drawing.type === "freehand") {
        return drawing.points
          .slice(1)
          .reduce(
            (distance, next, index) =>
              Math.min(
                distance,
                distanceToSegment(point, drawing.points[index], next),
              ),
            Number.POSITIVE_INFINITY,
          )
      }
      if (drawing.type === "text") {
        return Math.hypot(point.x - drawing.point.x, point.y - drawing.point.y)
      }
      const left = Math.min(drawing.start.x, drawing.end.x)
      const right = Math.max(drawing.start.x, drawing.end.x)
      const top = Math.min(drawing.start.y, drawing.end.y)
      const bottom = Math.max(drawing.start.y, drawing.end.y)
      if (drawing.type === "rectangle") {
        const corners = [
          { x: left, y: top },
          { x: right, y: top },
          { x: right, y: bottom },
          { x: left, y: bottom },
        ]
        return corners.reduce(
          (distance, corner, index) =>
            Math.min(
              distance,
              distanceToSegment(
                point,
                corner,
                corners[(index + 1) % corners.length],
              ),
            ),
          Number.POSITIVE_INFINITY,
        )
      }
      const center = { x: (left + right) / 2, y: (top + bottom) / 2 }
      const radiusX = (right - left) / 2
      const radiusY = (bottom - top) / 2
      return Math.min(
        ...Array.from({ length: 48 }, (_, index) => {
          const nextAngle = ((index + 1) / 48) * Math.PI * 2
          const angle = (index / 48) * Math.PI * 2
          const first = {
            x: center.x + Math.cos(angle) * radiusX,
            y: center.y + Math.sin(angle) * radiusY,
          }
          const second = {
            x: center.x + Math.cos(nextAngle) * radiusX,
            y: center.y + Math.sin(nextAngle) * radiusY,
          }
          return distanceToSegment(point, first, second)
        }),
      )
    }

    updateBoard((current) => {
      const candidates = [
        ...current.drawings.map((drawing) => ({
          id: drawing.id,
          type: "drawing" as const,
          distance: drawingDistance(drawing),
        })),
        ...current.arrows.map((arrow) => ({
          id: arrow.id,
          type: "arrow" as const,
          distance: distanceToSegment(point, arrow.start, arrow.end),
        })),
      ].sort((left, right) => left.distance - right.distance)
      const closest = candidates[0]
      if (!closest || closest.distance > 28) return current
      return closest.type === "drawing"
        ? {
            ...current,
            drawings: current.drawings.filter(
              (drawing) => drawing.id !== closest.id,
            ),
          }
        : {
            ...current,
            arrows: current.arrows.filter((arrow) => arrow.id !== closest.id),
          }
    })
  }

  function handlePitchClick(event: React.MouseEvent<HTMLDivElement>) {
    if (mode !== "arrow" && mode !== "eraser" && mode !== "text") return
    const rect = event.currentTarget.getBoundingClientRect()
    const point = {
      x: Math.max(
        3,
        Math.min(97, ((event.clientX - rect.left) / rect.width) * 100),
      ),
      y: Math.max(
        3,
        Math.min(97, ((event.clientY - rect.top) / rect.height) * 100),
      ),
    }
    if (mode === "eraser") {
      eraseAt(point)
      return
    }
    if (mode === "text") {
      const text = window.prompt("Enter annotation text")?.trim()
      if (!text) return
      updateBoard((current) => ({
        ...current,
        drawings: [
          ...current.drawings,
          {
            id: crypto.randomUUID(),
            type: "text",
            color: drawingColor,
            size: drawingSize,
            point,
            text: text.slice(0, 100),
          },
        ],
      }))
      return
    }
    if (!pendingArrow) {
      setPendingArrow(point)
      return
    }
    updateBoard((current) => ({
      ...current,
      arrows: [
        ...current.arrows,
        {
          id: crypto.randomUUID(),
          start: pendingArrow,
          end: point,
        },
      ],
    }))
    setPendingArrow(null)
  }

  function handlePitchPointerDown(event: React.PointerEvent<HTMLDivElement>) {
    if (mode !== "freehand" && mode !== "circle" && mode !== "rectangle") {
      return
    }
    event.preventDefault()
    const point = pointFromEvent(event)
    event.currentTarget.setPointerCapture(event.pointerId)
    setDraftDrawing(
      mode === "freehand"
        ? {
            id: crypto.randomUUID(),
            type: "freehand",
            color: drawingColor,
            size: drawingSize,
            points: [point],
          }
        : {
            id: crypto.randomUUID(),
            type: mode,
            color: drawingColor,
            size: drawingSize,
            start: point,
            end: point,
          },
    )
  }

  function handlePitchKeyDown(event: React.KeyboardEvent<HTMLDivElement>) {
    if (event.key === "Escape") {
      setPendingArrow(null)
    }
  }

  function handlePlayerSelect(player: Player, playerSide: TeamSide) {
    setSide(playerSide)
    setSelectedPlayerId(player.id)
  }

  function handlePlayerPointerDown(
    event: React.PointerEvent<HTMLButtonElement>,
    player: Player,
  ) {
    event.stopPropagation()
    if (mode !== "select") return
    svgRef.current?.setPointerCapture(event.pointerId)
    const playerSide = board.home.some((item) => item.id === player.id)
      ? "home"
      : "away"
    setSide(playerSide)
    setSelectedPlayerId(player.id)
    setDraggedPlayer({ side: playerSide, id: player.id })
  }

  function handlePlayerPointerMove(event: React.PointerEvent<HTMLDivElement>) {
    const point = pointFromEvent(event)
    if (draggedPlayer) {
      updateBoard((current) => ({
        ...current,
        [draggedPlayer.side]: current[draggedPlayer.side].map((player) =>
          player.id === draggedPlayer.id
            ? { ...player, x: point.x, y: point.y }
            : player,
        ),
      }))
      return
    }
    if (draftDrawing) {
      setDraftDrawing((current) => {
        if (!current) return null
        return current.type === "freehand"
          ? { ...current, points: [...current.points, point] }
          : { ...current, end: point }
      })
    }
  }

  function handlePitchPointerUp() {
    setDraggedPlayer(null)
    const drawing = draftDrawing
    setDraftDrawing(null)
    if (!drawing) return
    if (
      (drawing.type === "freehand" && drawing.points.length < 2) ||
      ("start" in drawing &&
        drawing.start.x === drawing.end.x &&
        drawing.start.y === drawing.end.y)
    ) {
      return
    }
    updateBoard((current) => ({
      ...current,
      drawings: [...current.drawings, drawing],
    }))
  }

  function handlePitchPointerCancel() {
    setDraggedPlayer(null)
    setDraftDrawing(null)
  }

  function exportBoard() {
    const svg = svgRef.current
    if (!svg) {
      throw new Error("The tactical pitch is not ready to export.")
    }
    const source = svg.cloneNode(true) as SVGSVGElement
    source.setAttribute("xmlns", "http://www.w3.org/2000/svg")
    const blob = new Blob([new XMLSerializer().serializeToString(source)], {
      type: "image/svg+xml;charset=utf-8",
    })
    const url = URL.createObjectURL(blob)
    const link = document.createElement("a")
    link.href = url
    link.download = `${projectName.replace(/[^a-z0-9-_]+/gi, "-") || "tactical-board"}.svg`
    link.click()
    URL.revokeObjectURL(url)
  }

  if (boardQuery.isPending) {
    return <p role="status">Loading board…</p>
  }

  if (boardQuery.isError) {
    return (
      <Alert variant="destructive">
        <AlertTitle>Could not load the tactical board</AlertTitle>
        <AlertDescription>{boardQuery.error.message}</AlertDescription>
      </Alert>
    )
  }

  return (
    <div className="flex flex-col gap-5">
      <ProjectCategories active="tactical-board" />
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <Link
            className="mb-2 inline-flex items-center gap-2 text-sm text-muted-foreground transition-colors hover:text-foreground"
            search={{ projectId: undefined, projectType }}
            to="/projects/tactical-board"
          >
            <ArrowLeft className="size-4" />
            All tactical projects
          </Link>
          <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">
            {projectName}
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Tactical Board
            {isDirty ? " · Unsaved changes" : " · All changes saved"}
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button
            onClick={() => saveMutation.mutate(board)}
            disabled={!isDirty || saveMutation.isPending}
          >
            <Save className="size-4" />
            {saveMutation.isPending ? "Saving…" : "Save board"}
          </Button>
          <Button onClick={exportBoard} variant="outline">
            <Download className="size-4" />
            Export SVG
          </Button>
        </div>
      </div>

      {saveMutation.isError && (
        <Alert variant="destructive">
          <AlertTitle>Board could not be saved</AlertTitle>
          <AlertDescription>{saveMutation.error.message}</AlertDescription>
        </Alert>
      )}

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_300px]">
        <Card className="min-w-0 border-border/80 bg-card/90">
          <CardHeader className="flex-row flex-wrap items-center justify-between gap-3">
            <div>
              <CardTitle className="text-base">Match plan</CardTitle>
              <p className="mt-1 text-sm text-muted-foreground">
                Drag players to reposition them
                {mode === "arrow"
                  ? " · Select two points to draw a movement"
                  : mode === "freehand"
                    ? " · Drag on the pitch to sketch"
                    : mode === "circle" || mode === "rectangle"
                      ? " · Drag to draw a shape"
                      : mode === "eraser"
                        ? " · Click a drawing or movement to erase it"
                        : ""}
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-1 rounded-lg border bg-background/60 p-1">
              <Button
                aria-pressed={mode === "select"}
                className="h-8"
                onClick={() => selectTool("select")}
                size="sm"
                variant={mode === "select" ? "secondary" : "ghost"}
              >
                Select
              </Button>
              <Button
                aria-pressed={mode === "arrow"}
                className="h-8"
                onClick={() => selectTool("arrow")}
                size="sm"
                variant={mode === "arrow" ? "secondary" : "ghost"}
              >
                <ArrowRight className="size-4" />
                Movement
              </Button>
              <Button
                aria-pressed={mode === "freehand"}
                className="h-8"
                onClick={() => selectTool("freehand")}
                size="sm"
                variant={mode === "freehand" ? "secondary" : "ghost"}
              >
                <Brush className="size-4" />
                Freehand
              </Button>
              <Button
                aria-pressed={mode === "circle"}
                className="h-8"
                onClick={() => selectTool("circle")}
                size="sm"
                variant={mode === "circle" ? "secondary" : "ghost"}
              >
                <Circle className="size-4" />
                Circle
              </Button>
              <Button
                aria-pressed={mode === "rectangle"}
                className="h-8"
                onClick={() => selectTool("rectangle")}
                size="sm"
                variant={mode === "rectangle" ? "secondary" : "ghost"}
              >
                <RectangleHorizontal className="size-4" />
                Rectangle
              </Button>
              <Button
                aria-pressed={mode === "text"}
                className="h-8"
                onClick={() => selectTool("text")}
                size="sm"
                variant={mode === "text" ? "secondary" : "ghost"}
              >
                <TextCursorInput className="size-4" />
                Text
              </Button>
              <Button
                aria-pressed={mode === "eraser"}
                className="h-8"
                onClick={() => selectTool("eraser")}
                size="sm"
                variant={mode === "eraser" ? "secondary" : "ghost"}
              >
                <Eraser className="size-4" />
                Eraser
              </Button>
              <label
                className="flex h-8 items-center gap-2 rounded-md px-2 text-xs text-muted-foreground"
                htmlFor="drawing-color"
              >
                Color
                <input
                  aria-label="Drawing color"
                  className="size-6 cursor-pointer rounded border-0 bg-transparent p-0"
                  id="drawing-color"
                  onChange={(event) => setDrawingColor(event.target.value)}
                  type="color"
                  value={drawingColor}
                />
              </label>
              <label
                className="flex h-8 items-center gap-2 rounded-md px-2 text-xs text-muted-foreground"
                htmlFor="drawing-size"
              >
                Size
                <input
                  className="w-20 accent-primary"
                  id="drawing-size"
                  max={100}
                  min={7}
                  onChange={(event) =>
                    setDrawingSize(Number(event.target.value))
                  }
                  type="range"
                  value={drawingSize}
                />
              </label>
            </div>
          </CardHeader>
          <CardContent>
            <TacticalPitch
              board={board}
              draftDrawing={draftDrawing}
              mode={mode}
              onPitchClick={handlePitchClick}
              onPitchKeyDown={handlePitchKeyDown}
              onPitchPointerCancel={handlePitchPointerCancel}
              onPitchPointerDown={handlePitchPointerDown}
              onPlayerSelect={handlePlayerSelect}
              onPlayerPointerDown={handlePlayerPointerDown}
              onPlayerPointerMove={handlePlayerPointerMove}
              onPlayerPointerUp={handlePitchPointerUp}
              pendingArrow={pendingArrow}
              selectedPlayerId={selectedPlayerId}
              side={side}
              svgRef={svgRef}
            />
            <p className="mt-3 text-xs text-muted-foreground">
              {board.home.length + board.away.length} players ·{" "}
              {board.arrows.length} movements · {board.drawings.length} drawings
            </p>
            {(board.drawings.length > 0 || board.arrows.length > 0) && (
              <div className="mt-3 flex flex-wrap gap-2">
                <Button
                  className="h-8"
                  disabled={board.drawings.length === 0}
                  onClick={() =>
                    updateBoard((current) => ({
                      ...current,
                      drawings: current.drawings.slice(0, -1),
                    }))
                  }
                  size="sm"
                  variant="outline"
                >
                  <ArrowLeft className="size-3.5" />
                  Undo drawing
                </Button>
                <Button
                  className="h-8"
                  onClick={() =>
                    updateBoard((current) => ({
                      ...current,
                      drawings: [],
                      arrows: [],
                    }))
                  }
                  size="sm"
                  variant="outline"
                >
                  <Trash2 className="size-3.5" />
                  Clear drawings
                </Button>
              </div>
            )}
            {board.arrows.length > 0 && (
              <fieldset
                aria-label="Movement controls"
                className="mt-3 flex flex-wrap gap-2 border-0 p-0"
              >
                {board.arrows.map((arrow, index) => (
                  <Button
                    aria-label={`Remove movement ${index + 1}`}
                    className="h-8"
                    key={arrow.id}
                    onClick={() =>
                      updateBoard((current) => ({
                        ...current,
                        arrows: current.arrows.filter(
                          (item) => item.id !== arrow.id,
                        ),
                      }))
                    }
                    size="sm"
                    variant="outline"
                  >
                    <Trash2 className="size-3.5" />
                    Movement {index + 1}
                  </Button>
                ))}
              </fieldset>
            )}
          </CardContent>
        </Card>

        <Card className="border-border/80 bg-card/90">
          <CardHeader>
            <CardTitle className="text-base">Squad &amp; tools</CardTitle>
          </CardHeader>
          <CardContent className="grid gap-5">
            <div className="grid grid-cols-2 gap-2 rounded-lg border bg-background/50 p-1">
              {(["home", "away"] as const).map((team) => (
                <Button
                  aria-pressed={side === team}
                  className="h-9"
                  key={team}
                  onClick={() => {
                    setSide(team)
                    setSelectedPlayerId(null)
                  }}
                  variant={side === team ? "secondary" : "ghost"}
                >
                  <span
                    className={`size-2.5 rounded-full ${
                      team === "home" ? "bg-primary" : "bg-slate-100"
                    }`}
                  />
                  {team === "home" ? "Home" : "Away"}
                  <span className="text-xs text-muted-foreground">
                    {board[team].length}
                  </span>
                </Button>
              ))}
            </div>

            <div className="grid gap-2">
              <Label htmlFor="team-formation">Formation</Label>
              <select
                className="h-9 rounded-md border border-input bg-background px-3 text-sm outline-none transition-colors focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/40"
                id="team-formation"
                onChange={(event) => updateFormation(event.target.value)}
                value={board.formations[side]}
              >
                {formations.map((formation) => (
                  <option key={formation} value={formation}>
                    {formation}
                  </option>
                ))}
              </select>
              <p className="text-xs text-muted-foreground">
                Applying a formation repositions this team’s players.
              </p>
            </div>

            <div className="flex items-center justify-between gap-2">
              <p className="text-sm font-medium">
                {side === "home" ? "Home squad" : "Away squad"}
              </p>
              <Button
                disabled={board[side].length >= 20}
                onClick={addPlayer}
                size="sm"
                variant="outline"
              >
                <Plus className="size-4" />
                Add player
              </Button>
            </div>

            <div className="max-h-56 space-y-1 overflow-y-auto pr-1">
              {board[side].map((player) => (
                <button
                  aria-pressed={selectedPlayerId === player.id}
                  className={`flex w-full items-center gap-3 rounded-md px-2 py-2 text-left text-sm transition-colors ${
                    selectedPlayerId === player.id
                      ? "bg-primary/10 text-foreground"
                      : "text-muted-foreground hover:bg-muted/70 hover:text-foreground"
                  }`}
                  key={player.id}
                  onClick={() => setSelectedPlayerId(player.id)}
                  type="button"
                >
                  <span className="flex size-7 shrink-0 items-center justify-center rounded-full border bg-background font-semibold text-foreground">
                    {player.number}
                  </span>
                  <span className="truncate">{player.name}</span>
                </button>
              ))}
              {board[side].length === 0 && (
                <p className="py-2 text-sm text-muted-foreground">
                  Add a player to start this squad.
                </p>
              )}
            </div>

            {selectedPlayer && (
              <div className="grid gap-3 rounded-lg border bg-background/50 p-3">
                <p className="text-sm font-medium">Selected player</p>
                <div className="grid grid-cols-[72px_1fr] gap-2">
                  <div className="grid gap-1.5">
                    <Label htmlFor="player-number">Number</Label>
                    <Input
                      id="player-number"
                      max={99}
                      min={0}
                      onChange={(event) =>
                        updatePlayer(selectedPlayer.id, {
                          number: Math.max(
                            0,
                            Math.min(99, Number(event.target.value) || 0),
                          ),
                        })
                      }
                      type="number"
                      value={selectedPlayer.number}
                    />
                  </div>
                  <div className="grid gap-1.5">
                    <Label htmlFor="player-name">Name</Label>
                    <Input
                      id="player-name"
                      maxLength={24}
                      onChange={(event) =>
                        updatePlayer(selectedPlayer.id, {
                          name: event.target.value,
                        })
                      }
                      value={selectedPlayer.name}
                    />
                  </div>
                </div>
                <Button
                  className="justify-start"
                  onClick={removeSelectedPlayer}
                  size="sm"
                  variant="ghost"
                >
                  <Trash2 className="size-4" />
                  Remove player
                </Button>
              </div>
            )}

            <div className="grid gap-2 border-t pt-4">
              <p className="text-sm font-medium">Board actions</p>
              <Button
                className="justify-start"
                onClick={() =>
                  updateBoard((current) => ({
                    ...current,
                    arrows: [],
                  }))
                }
                disabled={board.arrows.length === 0}
                size="sm"
                variant="outline"
              >
                <ArrowUp className="size-4" />
                Clear movement arrows
              </Button>
              <Button
                className="justify-start"
                onClick={() => updateFormation(board.formations[side])}
                size="sm"
                variant="ghost"
              >
                <ArrowDown className="size-4" />
                Reset {board.formations[side]} formation
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
