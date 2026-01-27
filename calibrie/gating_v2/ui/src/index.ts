/**
 * Calibrie Gating UI Extension for declair-ui.
 *
 * This module registers domain-specific data handlers for flow cytometry gating.
 * The widget implementations are now in declair-ui's generic primitives.
 */

import { registerFlowCytometryHandlers } from "./handlers/flow-cytometry";

// Register flow cytometry data preparation handlers
registerFlowCytometryHandlers();

// Re-export handlers for external use
export {
  registerFlowCytometryHandlers,
  flowCytometryHandler,
  flowCytometryDeckHandler,
  flowCytometryActiveHandler,
} from "./handlers/flow-cytometry";
