# Climate Veritas Frontend

Dashboard independiente construido con Next.js App Router, TypeScript,
Tailwind CSS, Lucide Icons y Recharts.

## Ejecución

Primero inicia la API FastAPI en `http://localhost:8000`. Después:

```bash
npm install
npm run dev
```

Abre `http://localhost:3000`.

Para cambiar la API, copia `.env.example` a `.env.local` y ajusta
`NEXT_PUBLIC_API_BASE_URL`.

## Rutas

- `/`: analizador de publicaciones y resultado neuro-simbólico.
- `/history`: historial persistido en PostgreSQL.
- `/experiments`: métricas, experimentos y estudio de ablación.
