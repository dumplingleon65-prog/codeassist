# Simulation Setup

## Architecture Overview

### AppWrapper Component
The `AppWrapper.tsx` conditionally renders based on environment:

```typescript
const isSimulationMode = process.env.NEXT_PUBLIC_SIMULATION_MODE === "true";

export default function AppWrapper() {
  if (isSimulationMode) {
    console.log("🤖 Starting in SIMULATION mode");
    return <SimulationApp />;
  } else {
    console.log("👤 Starting in NORMAL mode");
    return <App />;
  }
}
```

### Component Structure
```
web-ui/src/
├── App.tsx              # Normal mode (clean UI)
├── SimulationApp.tsx    # Simulation mode (identical UI + auto-start)
├── AppWrapper.tsx       # Environment-based switcher
├── simulation/          # Simulation logic
│   ├── scenarios.ts     # Scenario definitions
│   ├── hooks/          
│   ├── simulators/
│   └── types.ts
└── components/          # Shared components
```

## Usage

### Development Mode

```bash
cd web-ui

# Normal mode
npm run dev
# Access: http://localhost:3001

# Simulation mode  
npm run dev:simulation
# Access: http://localhost:3002
```

### Docker Compose

```bash
docker-compose up

# Both services use the same web-ui/ codebase:
# Normal UI: http://localhost:3000
# Simulation UI: http://localhost:3002
```

### Production Builds

```bash
# Normal build
npm run build && npm run start

# Simulation build
npm run build:simulation && npm run start:simulation
```

## Available Scripts

```json
{
  "dev": "next dev -p 3001",
  "dev:simulation": "NEXT_PUBLIC_SIMULATION_MODE=true next dev -p 3002",
  "build": "next build", 
  "build:simulation": "NEXT_PUBLIC_SIMULATION_MODE=true next build",
  "start": "next start -p 3001",
  "start:simulation": "NEXT_PUBLIC_SIMULATION_MODE=true next start -p 3002"
}
```

## Simulation Scenarios

### Scenario 1: Enter Periodically

Defined in `web-ui/src/simulation/scenarios.ts`:

```typescript
enter_periodically: {
  id: "enter_periodically",
  name: "Enter Periodically",
  description: "Human occasionally presses Enter but doesn't type code",
  config: {
    intervalMs: 3000,        // Press Enter every 3 seconds
    maxActions: 50,          // Maximum 50 Enter presses
    durationMs: 300000,      // Run for 5 minutes max
  },
}
```

### Adding New Scenarios

Simply edit `scenarios.ts`:

```typescript
export const SIMULATION_SCENARIOS: Record<string, SimulationScenario> = {
  enter_periodically: { /* existing */ },
  
  slow_typing: {
    id: "slow_typing",
    name: "Slow Typing", 
    description: "Types slowly with realistic pauses",
    config: {
      intervalMs: 1500,
      maxActions: 100,
      durationMs: 600000,
    },
  },
};
```

## Environment Variables

### Development
Create `.env.local` in `web-ui/`:

```env
# Force simulation mode
NEXT_PUBLIC_SIMULATION_MODE=true

# Override default scenario  
NEXT_PUBLIC_SIMULATION_SCENARIO=enter_periodically
```

### Docker
Set in `compose.yml`:

```yaml
simulation-ui:
  build:
    context: ./web-ui
    args:
      - NEXT_PUBLIC_SIMULATION_MODE=true
```


## Debugging

### Console Logs
Watch for mode detection:
```
👤 Starting in NORMAL mode      # Normal mode
🤖 Starting in SIMULATION mode  # Simulation mode
```

### Episode Verification


Both modes currently store episodes in:
```
persistent-data/state-service/episodes/
```

We can change this for the simulated episodes in the future
