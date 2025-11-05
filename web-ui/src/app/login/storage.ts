/**
 * JSON-backed storage for login state under {PERSISTENT_DATA_DIR}/auth (default: /app/persistent-data/auth).
 * Why: persisting org-level API keys and activation status for run script
 */
import { promises as fs } from "fs";
import path from "path";
import { z } from "zod";

const KeyEntry = z.object({
  publicKey: z.string(),
  privateKey: z.string(),
  createdAt: z.number(),
  activated: z.boolean(),
  activatedAt: z.number().optional(),
  accountAddress: z.string().optional(),
  initCode: z.string().optional(),
  deferredActionDigest: z.string().optional(),
});
export type KeyEntry = z.infer<typeof KeyEntry>;

const User = z.object({
  orgId: z.string(),
  updatedAt: z.number(),
  latestPublicKey: z.string().optional(),
  accountAddress: z.string().optional(),
});
export type User = z.infer<typeof User>;


function getPersistDir(): string {
  return process.env.PERSISTENT_DATA_DIR || "/app/persistent-data";
}

function authBaseDir(): string {
  return path.join(getPersistDir(), "auth");
}

// Single-file DB keyed by orgId
export type OrgRecord = { user: User; keys: KeyEntry[] };
export type DB = Record<string, OrgRecord>;

function dbPath(): string {
  return path.join(authBaseDir(), "userKeyMap.json");
}

async function ensureDir(p: string): Promise<void> {
  await fs.mkdir(p, { recursive: true });
}

export async function ensureAuthDir(): Promise<string> {
  const dir = authBaseDir();
  await ensureDir(dir);
  return dir;
}

async function readJson<T>(file: string, fallback: T): Promise<T> {
  try {
    const data = await fs.readFile(file, "utf8");
    return JSON.parse(data) as T;
  } catch (err: any) {
    if (err && (err.code === "ENOENT" || err.code === "ENOTDIR")) {
      return fallback;
    }
    throw err;
  }
}

async function readDB(): Promise<DB> {
  await ensureAuthDir();
  return readJson<DB>(dbPath(), {});
}

async function writeDB(db: DB): Promise<void> {
  await writeJson(dbPath(), db);
}

async function writeJson(file: string, data: unknown): Promise<void> {
  await ensureDir(path.dirname(file));
  await fs.writeFile(file, JSON.stringify(data, null, 2), "utf8");
}

export async function upsertUser(orgId: string, data: Omit<Partial<User>, "orgId" | "updatedAt">): Promise<User> {
  const db = await readDB();
  const now = Date.now();
  const org = db[orgId] ?? { user: { orgId, updatedAt: now }, keys: [] } as OrgRecord;
  const merged: User = User.parse({ ...org.user, ...data, orgId, updatedAt: now });
  db[orgId] = { ...org, user: merged };
  await writeDB(db);
  return merged;
}

export async function addKey(orgId: string, publicKey: string, privateKey: string): Promise<KeyEntry> {
  const db = await readDB();
  const now = Date.now();
  const org = db[orgId] ?? { user: { orgId, updatedAt: now }, keys: [] } as OrgRecord;
  const entry: KeyEntry = KeyEntry.parse({ publicKey, privateKey, createdAt: now, activated: false });
  const list = [...org.keys, entry];
  const user = User.parse({ ...org.user, latestPublicKey: publicKey, orgId, updatedAt: now });
  db[orgId] = { user, keys: list };
  await writeDB(db);
  return entry;
}

export async function getLatestKey(orgId: string): Promise<KeyEntry | null> {
  const db = await readDB();
  const list = db[orgId]?.keys ?? [];
  if (list.length === 0) return null;
  const latest = list.reduce((a, b) => (a.createdAt >= b.createdAt ? a : b));
  return KeyEntry.parse(latest);
}

export async function setKeyActivated(
  orgId: string,
  publicKey: string,
  data: { accountAddress: string; initCode: string; deferredActionDigest: string },
): Promise<KeyEntry | null> {
  const db = await readDB();
  const org = db[orgId];
  if (!org) return null;
  const idx = org.keys.findIndex((k) => k.publicKey === publicKey);
  if (idx < 0) return null;

  const updated: KeyEntry = KeyEntry.parse({
    ...org.keys[idx],
    activated: true,
    activatedAt: Date.now(),
    accountAddress: data.accountAddress,
    initCode: data.initCode,
    deferredActionDigest: data.deferredActionDigest,
  });

  const keys = [...org.keys];
  keys[idx] = updated;
  const user = User.parse({ ...org.user, latestPublicKey: publicKey, accountAddress: data.accountAddress, orgId });
  db[orgId] = { user, keys };
  await writeDB(db);
  return updated;
}

export async function hasUserRecord(orgId: string): Promise<boolean> {
  const db = await readDB();
  const org = db[orgId];
  return org !== undefined && org.keys.length > 0;
}

