import { app } from "../../scripts/app.js";

console.log("ComfyUI-QwenImageLoraLoader: JavaScript file loaded!");

let origProps = {};
let initialized = false;

const findWidgetByName = (node, name) => {
    const widget = node.widgets ? node.widgets.find((w) => w.name === name) : null;
    return widget;
};

const doesInputWithNameExist = (node, name) => {
    return false;
};

const HIDDEN_TAG = "tschide";

// Toggle Widget + change size
function toggleWidget(node, widget, show = false, suffix = "") {
    if (!widget || doesInputWithNameExist(node, widget.name)) return;

    if (!origProps[widget.name]) {
        origProps[widget.name] = { origType: widget.type, origComputeSize: widget.computeSize };
    }

    const newType = show ? origProps[widget.name].origType : HIDDEN_TAG + suffix;
    
    widget.type = newType;
    widget.computeSize = show ? origProps[widget.name].origComputeSize : () => [0, -4];

    widget.linkedWidgets?.forEach(w => toggleWidget(node, w, show, ":" + widget.name));

    const newHeight = node.computeSize()[1];
    node.setSize([node.size[0], newHeight + 35]);
}

// Handle multi-widget visibilities
function handleVisibility(node, countValue) {
    // Hide widgets beyond countValue (this fixes Issue #9)
    for (let i = countValue + 1; i <= 10; i++) {
        const nameWidget = findWidgetByName(node, `lora_name_${i}`);
        const strengthWidget = findWidgetByName(node, `lora_strength_${i}`);
        
        if (nameWidget) toggleWidget(node, nameWidget, false);
        if (strengthWidget) toggleWidget(node, strengthWidget, false);
    }

    // Show widgets up to countValue
    for (let i = 1; i <= countValue; i++) {
        const nameWidget = findWidgetByName(node, `lora_name_${i}`);
        const strengthWidget = findWidgetByName(node, `lora_strength_${i}`);
        
        if (nameWidget) toggleWidget(node, nameWidget, true);
        if (strengthWidget) toggleWidget(node, strengthWidget, true);
    }
}

// LoRA Count Handler
function handleLoRAStackerLoraCount(node, widget) {
    console.log("QwenImage LoRA Stack: lora_count changed to:", widget.value);
    handleVisibility(node, widget.value);
    
    // Multiple attempts with proper padding to ensure all widgets are visible
    [50, 150, 300].forEach((delay, index) => {
        setTimeout(() => {
            if (typeof node.setSize === 'function' && typeof node.computeSize === 'function') {
                const newHeight = node.computeSize()[1];
                // Base padding: 12px + 4px per attempt
                const padding = 12 + (index * 4);
                const paddedHeight = newHeight + padding;
                node.setSize([node.size[0], paddedHeight]);
                if (node.graph && typeof node.graph.setDirty === 'function') {
                    node.graph.setDirty(true, true);
                }
            }
        }, delay);
    });
}

// Map of node to widget handlers
const nodeWidgetHandlers = {
    "NunchakuQwenImageLoraStack": {
        'lora_count': handleLoRAStackerLoraCount
    }
};

function widgetLogic(node, widget) {
    const handler = nodeWidgetHandlers[node.comfyClass]?.[widget.name];
    if (handler) {
        handler(node, widget);
    }
}

app.registerExtension({
    name: "qwenimage.lorastack.widgethider",
    nodeCreated(node) {
        console.log("QwenImage LoRA Stack: Node created:", node.comfyClass);
        if (!nodeWidgetHandlers[node.comfyClass]) {
            return;
        }
        
        if (node.comfyClass === "NunchakuQwenImageLoraStack") {
            console.log("QwenImage LoRA Stack: Initializing node");
            
            // Debug: list all widget names
            if (node.widgets) {
                console.log("All widgets:", node.widgets.map(w => w.name).join(", "));
            }
            
            const loraCountWidget = findWidgetByName(node, "lora_count");
            
            if (loraCountWidget) {
                // Hide extra widgets beyond lora_count (fixes Issue #9)
                for (let i = loraCountWidget.value + 1; i <= 10; i++) {
                    const nameWidget = findWidgetByName(node, `lora_name_${i}`);
                    const strengthWidget = findWidgetByName(node, `lora_strength_${i}`);
                    if (nameWidget) toggleWidget(node, nameWidget, false);
                    if (strengthWidget) toggleWidget(node, strengthWidget, false);
                }
                
                // Initialize with current lora_count value
                handleVisibility(node, loraCountWidget.value);
                
                // Force initial height calculation with progressive padding
                [50, 100, 200, 500].forEach((delay, index) => {
                    setTimeout(() => {
                        if (typeof node.setSize === 'function' && typeof node.computeSize === 'function') {
                            const newHeight = node.computeSize()[1];
                            const padding = 20 + (index * 5);
                            const paddedHeight = newHeight + padding;
                            node.setSize([node.size[0], paddedHeight]);
                            if (node.graph && typeof node.graph.setDirty === 'function') {
                                node.graph.setDirty(true, true);
                            }
                        }
                    }, delay);
                });
            }
        }
        
        for (const w of node.widgets || []) {
            if (!nodeWidgetHandlers[node.comfyClass][w.name]) continue;
            let widgetValue = w.value;

            let originalDescriptor = Object.getOwnPropertyDescriptor(w, 'value');
            if (!originalDescriptor) {
                originalDescriptor = Object.getOwnPropertyDescriptor(w.constructor.prototype, 'value');
            }

            widgetLogic(node, w);

            Object.defineProperty(w, 'value', {
                get() {
                    let valueToReturn = originalDescriptor && originalDescriptor.get
                        ? originalDescriptor.get.call(w)
                        : widgetValue;

                    return valueToReturn;
                },
                set(newVal) {
                    if (originalDescriptor && originalDescriptor.set) {
                        originalDescriptor.set.call(w, newVal);
                    } else {
                        widgetValue = newVal;
                    }

                    widgetLogic(node, w);
                }
            });
        }
        setTimeout(() => {initialized = true;}, 500);
    }
});
