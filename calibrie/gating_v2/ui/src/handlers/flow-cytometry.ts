/**
 * Flow cytometry data preparation handlers for calibrie gating UI.
 * Uses createChannelHandler factory for declarative configuration.
 */

import {
  registerHandler,
  createChannelHandler,
} from "declair-ui-react/lib/dataPreparation";

const flowCytometryHandler = createChannelHandler({
  name: "flow-cytometry",
  channelFields: ["xaxis", "yaxis"],
  itemsField: "gates",
  selectionField: "selected_gate_index",
  mode: "single",
});

const flowCytometryDeckHandler = createChannelHandler({
  name: "flow-cytometry-deck",
  channelFields: ["xaxis", "yaxis"],
  itemsField: "gates",
  mode: "deck",
});

const flowCytometryActiveHandler = createChannelHandler({
  name: "flow-cytometry-active",
  channelFields: ["xaxis", "yaxis"],
  itemsField: "gates",
  selectionField: "selected_gate_index",
  mode: "active",
});

export function registerFlowCytometryHandlers(): void {
  registerHandler(flowCytometryHandler);
  registerHandler(flowCytometryDeckHandler);
  registerHandler(flowCytometryActiveHandler);
}

export {
  flowCytometryHandler,
  flowCytometryDeckHandler,
  flowCytometryActiveHandler,
};
