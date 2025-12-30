import { app } from "../../scripts/app.js";

console.log("★★★ zimageturbo_lora_dynamic.js: Z-Image-Turbo LoRA Stack V3 ★★★");

const HIDDEN_TAG = "tschide";

app.registerExtension({
    name: "nunchaku.zimageturbo_lora_dynamic_v3",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "NunchakuZImageTurboLoraStackV3") {
            nodeType["@visibleLoraCount"] = { type: "number", default: 1, min: 1, max: 10, step: 1 };
        }
    },

    nodeCreated(node) {
        if (node.comfyClass !== "NunchakuZImageTurboLoraStackV3") return;

        if (!node.properties) node.properties = {};
        if (node.properties["visibleLoraCount"] === undefined) node.properties["visibleLoraCount"] = 1;

        // Immediately hide lora_count widget if it exists
        const initialLoraCountWidget = node.widgets?.find(w => w.name === "lora_count");
        if (initialLoraCountWidget) {
            if (!initialLoraCountWidget.origType) {
                initialLoraCountWidget.origType = initialLoraCountWidget.type;
                initialLoraCountWidget.origComputeSize = initialLoraCountWidget.computeSize;
            }
            initialLoraCountWidget.type = HIDDEN_TAG;
            initialLoraCountWidget.computeSize = () => [0, -4];
        }

        node.cachedWidgets = {};
        let cacheReady = false;

        const initCache = () => {
            if (cacheReady) return;
            const all = [...node.widgets];

            // Cache lora_count widget (required for Python backend, but hidden in UI)
            const loraCountWidget = all.find(w => w.name === "lora_count");
            if (loraCountWidget) {
                node.cachedLoraCount = loraCountWidget;
                // Store original properties for restoration if needed
                if (!loraCountWidget.origType) {
                    loraCountWidget.origType = loraCountWidget.type;
                    loraCountWidget.origComputeSize = loraCountWidget.computeSize;
                }
                // Hide V1's lora_count widget using HIDDEN_TAG and computeSize
                loraCountWidget.type = HIDDEN_TAG;
                loraCountWidget.computeSize = () => [0, -4];
            }

            // Cache cpu_offload widget
            const cpuOffloadWidget = all.find(w => w.name === "cpu_offload");
            if (cpuOffloadWidget) {
                node.cachedCpuOffload = cpuOffloadWidget;
            }

            // Cache toggle_all widget
            const toggleAllWidget = all.find(w => w.name === "toggle_all");
            if (toggleAllWidget) {
                node.cachedToggleAll = toggleAllWidget;
            }

            for (let i = 1; i <= 10; i++) {
                const wEnabled = all.find(w => w.name === `enabled_${i}`);
                const wName = all.find(w => w.name === `lora_name_${i}`);
                const wStrength = all.find(w => w.name === `lora_strength_${i}`);
                if (wEnabled && wName && wStrength) {
                    node.cachedWidgets[i] = [wEnabled, wName, wStrength];
                    wEnabled.type = "toggle";
                    wName.type = "combo";
                    wStrength.type = "number";
                    if (wEnabled.computeSize) delete wEnabled.computeSize;
                    if (wName.computeSize) delete wName.computeSize;
                    if (wStrength.computeSize) delete wStrength.computeSize;
                }
            }
            cacheReady = true;
        };



        const ensureControlWidget = () => {
            const name = "🔢 LoRA Count";

            // Remove old button widgets
            for (let i = node.widgets.length - 1; i >= 0; i--) {
                const w = node.widgets[i];
                if (w.name === "🔢 Set LoRA Count" || w.type === "button") {
                    node.widgets.splice(i, 1);
                }
            }

            let w = node.widgets.find(x => x.name === name);
            if (!w) {
                const values = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"];
                w = node.addWidget("combo", name, "1", (v) => {
                    const num = parseInt(v);
                    if (!isNaN(num)) {
                        node.properties["visibleLoraCount"] = num;
                        if (node.cachedLoraCount) {
                            node.cachedLoraCount.value = num;
                        }
                        node.updateLoraSlots();
                    }
                }, { values });
            }
            w.value = node.properties["visibleLoraCount"].toString();
            if (node.cachedLoraCount) {
                node.cachedLoraCount.value = node.properties["visibleLoraCount"];
            }
            return w;
        };

        const ensureToggleAllWidget = () => {
            if (!node.cachedToggleAll) return null;
            if (!node.cachedToggleAll.origCallback) {
                node.cachedToggleAll.origCallback = node.cachedToggleAll.callback;
            }
            node.cachedToggleAll.callback = (value) => {
                if (node.cachedToggleAll.origCallback) {
                    node.cachedToggleAll.origCallback(value);
                }
                const count = parseInt(node.properties["visibleLoraCount"] || 1);
                for (let i = 1; i <= count; i++) {
                    const pair = node.cachedWidgets[i];
                    if (pair && pair[0]) {
                        pair[0].value = value;
                    }
                }
            };
            return node.cachedToggleAll;
        };

        // Plain logic restored
        node.updateLoraSlots = function () {
            if (!cacheReady) initCache();

            const count = parseInt(this.properties["visibleLoraCount"] || 1);
            const controlWidget = ensureControlWidget();

            this.widgets = [controlWidget];

            if (node.cachedLoraCount) {
                node.cachedLoraCount.type = HIDDEN_TAG;
                node.cachedLoraCount.computeSize = () => [0, -4];
                node.cachedLoraCount.value = count;
                this.widgets.push(node.cachedLoraCount);
            }

            const toggleAllWidget = ensureToggleAllWidget();
            if (toggleAllWidget) {
                this.widgets.push(toggleAllWidget);
            }

            if (node.cachedCpuOffload) {
                this.widgets.push(node.cachedCpuOffload);
            }

            for (let i = 1; i <= count; i++) {
                const pair = this.cachedWidgets[i];
                if (pair && pair.length >= 3) {
                    const wEnabled = pair[0];
                    const wName = pair[1];
                    const wStrength = pair[2];

                    // Restore standard widgets
                    wEnabled.type = "toggle";
                    wName.type = "combo";
                    wStrength.type = "number";

                    // Clear custom compute sizes if any
                    if (wEnabled.computeSize) delete wEnabled.computeSize;
                    if (wName.computeSize) delete wName.computeSize;
                    if (wStrength.computeSize) delete wStrength.computeSize;

                    this.widgets.push(wEnabled);
                    this.widgets.push(wName);
                    this.widgets.push(wStrength);
                }
            }

            const HEADER_H = 60;
            // 3 lines per slot (Toggle ~20 + Name ~26 + Strength ~26 + margins)
            const SLOT_H = 80;
            const TOGGLE_ALL_H = toggleAllWidget ? 30 : 0;
            const CPU_OFFLOAD_H = node.cachedCpuOffload ? 30 : 0;
            const PADDING = 20;
            const targetH = HEADER_H + TOGGLE_ALL_H + CPU_OFFLOAD_H + (count * SLOT_H) + PADDING;

            this.setSize([this.size[0], targetH]);

            if (app.canvas) app.canvas.setDirty(true, true);
        };

        node.onPropertyChanged = function (property, value) {
            if (property === "visibleLoraCount") {
                const w = this.widgets.find(x => x.name === "🔢 LoRA Count");
                if (w) w.value = value.toString();
                this.updateLoraSlots();
            }
        };

        const origOnConfigure = node.onConfigure;
        node.onConfigure = function () {
            if (origOnConfigure) origOnConfigure.apply(this, arguments);
            setTimeout(() => node.updateLoraSlots(), 100);
        };

        setTimeout(() => {
            initCache();
            node.updateLoraSlots();
        }, 100);
    }
});

