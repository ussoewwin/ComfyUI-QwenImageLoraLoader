import { app } from "../../scripts/app.js";

console.log("★★★ z_qwen_lora_dynamic.js: Qwen Image LoRA Stack V2 ★★★");

const HIDDEN_TAG = "tschide";

app.registerExtension({
    name: "nunchaku.qwen_lora_dynamic_combo_final_restore",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "NunchakuQwenImageLoraStackV2") {
            nodeType["@visibleLoraCount"] = { type: "number", default: 1, min: 1, max: 10, step: 1 };
        }
    },

        nodeCreated(node) {
        if (node.comfyClass !== "NunchakuQwenImageLoraStackV2") return;

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
            
            for (let i = 1; i <= 10; i++) {
                const wName = all.find(w => w.name === `lora_name_${i}`);
                const wStrength = all.find(w => w.name === `lora_strength_${i}`);
                if (wName && wStrength) {
                    node.cachedWidgets[i] = [wName, wStrength];
                    wName.type = "combo";
                    wStrength.type = "number";
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
                        // Sync with lora_count widget for Python backend
                        if (node.cachedLoraCount) {
                            node.cachedLoraCount.value = num;
                        }
                        node.updateLoraSlots();
                    }
                }, { values });
            }
            w.value = node.properties["visibleLoraCount"].toString();
            // Sync lora_count widget value
            if (node.cachedLoraCount) {
                node.cachedLoraCount.value = node.properties["visibleLoraCount"];
            }
            return w;
        };

        node.updateLoraSlots = function() {
            if (!cacheReady) initCache();

            const count = parseInt(this.properties["visibleLoraCount"] || 1);
            const controlWidget = ensureControlWidget();
        
            // Physical widget reconstruction for clean layout (like Flux V2)
            this.widgets = [controlWidget];

            // Add lora_count widget (required for Python backend, but hidden in UI using HIDDEN_TAG)
            if (node.cachedLoraCount) {
                // Ensure it's hidden
                node.cachedLoraCount.type = HIDDEN_TAG;
                node.cachedLoraCount.computeSize = () => [0, -4];
                // Sync value for Python backend
                node.cachedLoraCount.value = count;
                // Add to widgets array (for Python backend, but hidden in UI)
                this.widgets.push(node.cachedLoraCount);
            }

            // Add cpu_offload widget from cache (required for Python backend)
            if (node.cachedCpuOffload) {
                this.widgets.push(node.cachedCpuOffload);
            }

            // Add only visible LoRA slots (non-visible widgets are removed from array)
            for (let i = 1; i <= count; i++) {
                const pair = this.cachedWidgets[i];
                if (pair) {
                    this.widgets.push(pair[0]); 
                    this.widgets.push(pair[1]);
                }
            }

            // Height calculation
            const HEADER_H = 60;
            const SLOT_H = 54;
            const CPU_OFFLOAD_H = node.cachedCpuOffload ? 40 : 0;
            const PADDING = 20;
            const targetH = HEADER_H + CPU_OFFLOAD_H + (count * SLOT_H) + PADDING;
            
            this.setSize([this.size[0], targetH]);
            
            if (app.canvas) app.canvas.setDirty(true, true);
        };

        node.onPropertyChanged = function(property, value) {
            if (property === "visibleLoraCount") {
                const w = this.widgets.find(x => x.name === "🔢 LoRA Count");
                if (w) w.value = value.toString();
                this.updateLoraSlots();
            }
        };
        
        // Restore UI on configure
        const origOnConfigure = node.onConfigure;
        node.onConfigure = function() {
             if (origOnConfigure) origOnConfigure.apply(this, arguments);
             setTimeout(() => node.updateLoraSlots(), 100);
        };

        setTimeout(() => {
            initCache();
            node.updateLoraSlots();
        }, 100);
    }
});

