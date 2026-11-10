// Compiled from src/sessions.ts by scripts/docs.ts; edit the source, then run `make docs`.
/**
 * Session creation with backend fallback, shared by `VelaDecision` and `VelaTask`.
 *
 * The runtime adapters list the backends to try (`node.ts`: WebGPU, then CPU; `web.ts`: WebGPU, then WASM); this
 * module loads every graph of a bundle on the first backend that accepts all of them. WHY the whole bundle on one
 * backend: the image encoder's features feed the task graph, and a bundle split across GPU and CPU would copy them
 * between devices for no gain. WHY fall back at all: the GPU path is 3× faster where it works (VERIFIED: 337 ms
 * instead of 1,044 ms for 1,800 text tokens and 341 ms instead of 1,016 ms for one image, Node.js on an Apple
 * M-series Mac, 2026-11-08), but many machines and browsers have no usable WebGPU adapter. The backend that ran is
 * reported in every response and as `fallback` progress events, so the switch is never silent.
 * @module
 */
/**
 * Create the sessions of `graphs` on the first backend of `backends` that loads all of them.
 *
 * A backend that fails releases the sessions it already created before the next one is tried, so a failed attempt
 * leaks no weights.
 * @param graphs Graph name (for progress events) and reference, in load order.
 * @returns The backend used and one session per graph, in the order of `graphs`.
 * @throws The last backend's error when no backend loads the bundle; a `TypeError` when `backends` is empty.
 */
export async function createSessions(graphs, { createSession, backends, onProgress }) {
    if (!backends.length)
        throw new TypeError("No backend to load the bundle on");
    let error;
    for (const [i, backend] of backends.entries()) {
        const sessions = [];
        try {
            for (const [name, graph] of graphs) {
                onProgress({ stage: "load", graph: name, backend });
                sessions.push(await createSession(graph, backend));
            }
            return { backend, sessions };
        }
        catch (e) {
            error = e;
            await Promise.allSettled(sessions.map((s) => s.release()));
            const next = backends[i + 1];
            if (next)
                onProgress({ stage: "fallback", from: backend, to: next, error: e instanceof Error ? e.message : String(e) });
        }
    }
    throw error;
}
