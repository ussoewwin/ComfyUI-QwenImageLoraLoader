import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

console.log("★★★ zimageturbo_lora_dynamic_v1.js: Z-Image-Turbo LoRA Stack V1 ★★★");

// Use LiteGraph built-in colors to ensure Light/Dark Mode compatibility
const getColors = () => ({
  bg: LiteGraph.WIDGET_BGCOLOR,
  text: LiteGraph.WIDGET_TEXT_COLOR,
  secondary: LiteGraph.WIDGET_SECONDARY_TEXT_COLOR,
  outline: LiteGraph.WIDGET_OUTLINE_COLOR,
  active: LiteGraph.NODE_SELECTED_TITLE_COLOR,
});

const WIDGET_HEIGHT = 24;
const HORIZ_MARGIN = 15;

class NunchakuLoraWidget {
  constructor(name, parentNode) {
    this.name = name;
    this.type = "custom";
    this.parentNode = parentNode;
    this.value = { enabled: true, lora_name: "None", lora_strength: 1.0 };
    this.last_y = 0;

    // Define hit areas
    this.hitAreas = {
      toggle: [0, 0],
      lora: [0, 0],
      strengthDec: [0, 0],
      strengthInc: [0, 0],
      strengthVal: [0, 0]
    };
  }

  draw(ctx, node, width, posY, height) {
    const colors = getColors();
    const x = HORIZ_MARGIN;
    const midY = posY + height / 2;
    this.last_y = posY;
    const widgetWidth = width - HORIZ_MARGIN * 2;

    // --- Check global stack enabled state ---
    const globalOn = node.widgets.find(w => w.name === "stack_enabled")?.value !== false;
    const effectiveOn = this.value.enabled && globalOn;

    ctx.save();

    // 1. Background - using theme colors
    ctx.fillStyle = colors.bg;
    ctx.strokeStyle = colors.outline;
    ctx.beginPath();
    ctx.roundRect(x, posY + 2, widgetWidth, height - 4, [4]);
    ctx.fill();
    ctx.stroke();

    // 2. Toggle switch
    const toggleWidth = 20;
    const toggleHeight = 12;
    const toggleX = x + 8;
    const toggleY = midY - (toggleHeight / 2);

    // Track background
    ctx.fillStyle = colors.outline;
    ctx.beginPath();
    ctx.roundRect(toggleX, toggleY, toggleWidth, toggleHeight, [toggleHeight / 2]);
    ctx.fill();

    // Knob
    ctx.fillStyle = globalOn ? colors.text : colors.secondary; // Knob follows global state
    ctx.beginPath();
    // Position knob with smaller offsets to stay within the compact track
    const knobX = this.value.enabled ? toggleX + toggleWidth - 6 : toggleX + 6;
    ctx.arc(knobX, midY, 4, 0, Math.PI * 2); // Radius reduced to 4
    ctx.fill();
    this.hitAreas.toggle = [toggleX, toggleX + toggleWidth];

    // 3. LoRA Name (auto-calculate remaining width)
    const strengthAreaWidth = 70;
    const labelX = toggleX + toggleWidth + 10;
    const labelWidth = widgetWidth - (labelX - x) - strengthAreaWidth - 5;

    ctx.fillStyle = effectiveOn ? colors.text : colors.secondary;
    ctx.font = "12px Arial";
    ctx.textAlign = "left";
    let displayLabel = (this.value.lora_name || "None");

    // Text clipping
    const metrics = ctx.measureText(displayLabel);
    if (metrics.width > labelWidth) {
      const avgCharWidth = 7;
      const maxChars = Math.floor(labelWidth / avgCharWidth);
      if (displayLabel.length > maxChars) {
        displayLabel = displayLabel.substring(0, maxChars - 3) + "...";
      }
    }
    ctx.fillText(displayLabel, labelX, midY + 4);
    this.hitAreas.lora = [labelX, labelX + labelWidth];

    // 4. Strength control
    const rightX = x + widgetWidth - 5;
    const valueCenter = rightX - 35; // Center point for the numeric value
    const arrowGap = 22;            // Distance from center to arrows

    ctx.textAlign = "center";

    // ▶ symbol
    ctx.fillText("▶", valueCenter + arrowGap, midY + 4);
    this.hitAreas.strengthInc = [valueCenter + arrowGap - 10, valueCenter + arrowGap + 10];

    // Numeric value
    ctx.fillStyle = effectiveOn ? colors.text : colors.secondary;
    const valStr = this.value.lora_strength.toFixed(2);
    ctx.fillText(valStr, valueCenter, midY + 4);
    this.hitAreas.strengthVal = [valueCenter - 18, valueCenter + 18];

    // ◀ symbol
    ctx.fillText("◀", valueCenter - arrowGap, midY + 4);
    this.hitAreas.strengthDec = [valueCenter - arrowGap - 10, valueCenter - arrowGap + 10];

    ctx.restore();
  }

  mouse(event, pos, node) {
    if (event.type !== "pointerdown") return false;
    if (event.button === 2) return false; // Right-click is handled by the node

    // --- Block interaction if global stack enabled is off ---
    const globalOn = node.widgets.find(w => w.name === "stack_enabled")?.value !== false;
    if (!globalOn) return false;

    const widgetX = pos[0];
    // Determine hit area
    if (widgetX >= this.hitAreas.toggle[0] && widgetX <= this.hitAreas.toggle[1]) {
      this.value.enabled = !this.value.enabled;
    } else if (widgetX >= this.hitAreas.lora[0] && widgetX <= this.hitAreas.lora[1]) {
      this.chooseLora(event);
    } else if (widgetX >= this.hitAreas.strengthDec[0] && widgetX <= this.hitAreas.strengthDec[1]) {
      this.value.lora_strength = Math.round((this.value.lora_strength - 0.05) * 100) / 100;
    } else if (widgetX >= this.hitAreas.strengthInc[0] && widgetX <= this.hitAreas.strengthInc[1]) {
      this.value.lora_strength = Math.round((this.value.lora_strength + 0.05) * 100) / 100;
    } else if (widgetX >= this.hitAreas.strengthVal[0] && widgetX <= this.hitAreas.strengthVal[1]) {
      // Use ComfyUI built-in prompt
      app.canvas.prompt("Lora Strength", this.value.lora_strength, (v) => {
        const val = parseFloat(v);
        if (!isNaN(val)) {
          this.value.lora_strength = val;
          node.setDirtyCanvas(true);
        }
      }, event);
    } else {
      return false;
    }

    this.parentNode.setDirtyCanvas(true, true);
    return true;
  }

  async chooseLora(event) {
    // Get ComfyUI LoRA list
    const resp = await api.fetchApi("/object_info");
    const data = await resp.json();
    const loras = data?.LoraLoader?.input?.required?.lora_name?.[0] || [];

    // Use LiteGraph's ContextMenu
    const menu = new LiteGraph.ContextMenu(loras, {
      event: event,
      callback: (value) => {
        this.value.lora_name = value;
        this.parentNode.setDirtyCanvas(true, true);
      }
    });
  }

  serializeValue() {
    return this.value;
  }
}

app.registerExtension({
  name: "nunchaku.zimageturbo_lora_dynamic_v1",

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name !== "NunchakuZImageTurboLoraStackV1") return;

    // Save original onNodeCreated
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);
      // Inform ComfyUI to include widget values in the prompt sent to the backend
      this.serialize_widgets = true;
      this.loraWidgets = [];

      // Monitor stack_enabled switch
      const stackEnabledWidget = this.widgets.find(w => w.name === "stack_enabled");
      if (stackEnabledWidget) {
        stackEnabledWidget.callback = () => { this.setDirtyCanvas(true); };
      }

      // Add "Add Lora" button
      const btn = this.addWidget("button", "➕ Add Lora", null, (val, canvas, node, pos, event) => {
        this.addLoraRowWithChooser(event);
      });

      // Prevent the button from being serialized and sent to the backend
      btn.serialize = false;

      const HEADER_H = 60;
      const INPUT_WIDGETS_H = stackEnabledWidget ? 50 : 0;
      const PADDING = 20;
      const targetH = HEADER_H + INPUT_WIDGETS_H + PADDING;
      this.setSize([this.size[0], targetH]);
      if (app.canvas) app.canvas.setDirty(true, true);
    };

    // Show LoRA menu before adding the row
    nodeType.prototype.addLoraRowWithChooser = async function (event) {
      // Fetch LoRA list from ComfyUI object info
      const resp = await api.fetchApi("/object_info");
      const data = await resp.json();
      const loras = data?.LoraLoader?.input?.required?.lora_name?.[0] || [];

      // Show the selection menu
      new LiteGraph.ContextMenu(loras, {
        event: event,
        callback: (value) => {
          // Add the row only after a LoRA is selected
          this.addLoraRow({ enabled: true, lora_name: value, lora_strength: 1.0 });
        }
      });
    };

    // Add a new LoRA row
    nodeType.prototype.addLoraRow = function (data = null) {
      const name = `lora_${this.loraWidgets.length + 1}`;
      const w = new NunchakuLoraWidget(name, this);
      // Ensure we have a default value if no data is provided
      w.value = data || { enabled: true, lora_name: "None", lora_strength: 1.0 };

      // Insert above the button
      const btnIdx = this.widgets.findIndex(x => x.name === "➕ Add Lora");
      this.widgets.splice(btnIdx, 0, w);
      this.loraWidgets.push(w);

      this.setSize([this.size[0], this.computeSize()[1]]);
      this.setDirtyCanvas(true, true);
      return w;
    };

    // Right-click menu logic
    // Override getSlotInPosition to detect if a widget was clicked
    nodeType.prototype.getSlotInPosition = function (canvasX, canvasY) {
      const slot = LGraphNode.prototype.getSlotInPosition.apply(this, arguments);
      if (slot) return slot;

      // If no slot was clicked, check if the click was on our LoraWidget
      const widget = this.widgets.find(w => {
        return w instanceof NunchakuLoraWidget &&
          canvasY > (this.pos[1] + w.last_y) &&
          canvasY < (this.pos[1] + w.last_y + WIDGET_HEIGHT);
      });

      if (widget) {
        // Return dummy slot info, which triggers LiteGraph to call getSlotMenuOptions
        return { widget: widget, output: { type: "LORA_WIDGET" } };
      }
      return null;
    };

    // Show custom menu when the dummy slot is detected
    nodeType.prototype.getSlotMenuOptions = function (slot) {
      if (slot && slot.widget instanceof NunchakuLoraWidget) {
        const widget = slot.widget;
        return [
          {
            content: widget.value.enabled ? "⚫ Toggle Off" : "🟢 Toggle On",
            callback: () => {
              widget.value.enabled = !widget.value.enabled;
              this.setDirtyCanvas(true);
            }
          },
          {
            content: "⬆️ Move Up",
            callback: () => this.moveLora(widget, -1)
          },
          {
            content: "⬇️ Move Down",
            callback: () => this.moveLora(widget, 1)
          },
          {
            content: "🗑️ Remove",
            callback: () => {
              const idx = this.widgets.indexOf(widget);
              this.widgets.splice(idx, 1);
              this.loraWidgets = this.loraWidgets.filter(lw => lw !== widget);
              this.setSize([this.size[0], this.computeSize()[1]]);
              this.setDirtyCanvas(true);
            }
          }
        ];
      }
      return LGraphNode.prototype.getSlotMenuOptions?.apply(this, arguments);
    };

    nodeType.prototype.moveLora = function (widget, dir) {
      const idx = this.widgets.indexOf(widget);
      const targetIdx = idx + dir;
      // Ensure move range stays between LoraWidgets (prevent moving past or onto model/cpu_offload widgets)
      if (this.widgets[targetIdx] instanceof NunchakuLoraWidget) {
        this.widgets.splice(idx, 1);
        this.widgets.splice(targetIdx, 0, widget);
        this.setDirtyCanvas(true);
      }
    };

    // serialized data loading
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      onConfigure?.apply(this, arguments);
      if (info.widgets_values) {
        // Clear initialized widgets
        this.widgets = this.widgets.filter(w => !(w instanceof NunchakuLoraWidget));
        this.loraWidgets = [];

        // Recover LoRA rows from serialized data
        // ComfyUI's serialized data may exist in array format
        info.widgets_values.forEach((val, idx) => {
          if (val && typeof val === 'object' && val.lora_name !== undefined) {
            this.addLoraRow(val);
          }
        });

        // Make sure the "Add Lora" button stays at the bottom.
        if (this.addLoraBtn) {
          const btnIdx = this.widgets.indexOf(this.addLoraBtn);
          this.widgets.splice(btnIdx, 1);
          this.widgets.push(this.addLoraBtn);
        }
      }
    };
  }
});
