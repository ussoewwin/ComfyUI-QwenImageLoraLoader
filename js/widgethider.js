import { app } from "../../scripts/app.js";

console.log("ComfyUI-QwenImageLoraLoader: JavaScript file loaded!");

let origProps = {};
let initialized = false;

const findWidgetByName = (node, name) => {
    return node.widgets ? node.widgets.find((w) => w.name === name) : null;
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

    widget.type = show ? origProps[widget.name].origType : HIDDEN_TAG + suffix;
    widget.computeSize = show ? origProps[widget.name].origComputeSize : () => [0, -4];

    widget.linkedWidgets?.forEach(w => toggleWidget(node, w, show, ":" + widget.name));

    const newHeight = node.computeSize()[1];
    node.setSize([node.size[0], newHeight]);
}

// Handle multi-widget visibilities
function handleVisibility(node, countValue) {
    // First hide ALL widgets
    for (let i = 1; i <= 10; i++) {
        const nameWidget = findWidgetByName(node, `lora_name_${i}`);
        const strengthWidget = findWidgetByName(node, `lora_strength_${i}`);
        
        if (nameWidget) toggleWidget(node, nameWidget, false);
        if (strengthWidget) toggleWidget(node, strengthWidget, false);
    }

    // Show only the widgets up to lora_count
    for (let i = 1; i <= countValue; i++) {
        const nameWidget = findWidgetByName(node, `lora_name_${i}`);
        const strengthWidget = findWidgetByName(node, `lora_strength_${i}`);
        
        if (nameWidget) toggleWidget(node, nameWidget, true);
        if (strengthWidget) toggleWidget(node, strengthWidget, true);
    }
    
    // Final height update after all widgets are processed
    if (typeof node.setSize === 'function' && typeof node.computeSize === 'function') {
        const newHeight = node.computeSize()[1];
        const paddedHeight = newHeight + 30;
        node.setSize([node.size[0], paddedHeight]);
        
        if (node.graph && typeof node.graph.setDirty === 'function') {
            node.graph.setDirty(true, true);
        }
        
        // Additional height adjustment with multiple attempts
        [10, 50, 100, 200].forEach(delay => {
            setTimeout(() => {
                const finalHeight = node.computeSize()[1];
                const finalPaddedHeight = finalHeight + 30;
                if (finalPaddedHeight !== node.size[1]) {
                    node.setSize([node.size[0], finalPaddedHeight]);
                    if (node.graph && typeof node.graph.setDirty === 'function') {
                        node.graph.setDirty(true, true);
                    }
                }
            }, delay);
        });
    }
}

// LoRA Count Handler
function handleLoRAStackerLoraCount(node, widget) {
    console.log("QwenImage LoRA Stack: lora_count changed to:", widget.value);
    handleVisibility(node, widget.value);
    
    setTimeout(() => {
        if (typeof node.setSize === 'function' && typeof node.computeSize === 'function') {
            const newHeight = node.computeSize()[1];
            node.setSize([node.size[0], newHeight]);
            if (node.graph && typeof node.graph.setDirty === 'function') {
                node.graph.setDirty(true, true);
            }
        }
    }, 50);
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
            const loraCountWidget = findWidgetByName(node, "lora_count");
            
            if (loraCountWidget) {
                // First, hide all widgets to get a clean state
                for (let i = 1; i <= 10; i++) {
                    const nameWidget = findWidgetByName(node, `lora_name_${i}`);
                    const strengthWidget = findWidgetByName(node, `lora_strength_${i}`);
                    if (nameWidget) toggleWidget(node, nameWidget, false);
                    if (strengthWidget) toggleWidget(node, strengthWidget, false);
                }
                
                // Initialize with current lora_count value
                handleVisibility(node, loraCountWidget.value);
                
                // Force initial height calculation with multiple attempts
                [50, 100, 200, 500].forEach(delay => {
                    setTimeout(() => {
                        if (typeof node.setSize === 'function' && typeof node.computeSize === 'function') {
                            const newHeight = node.computeSize()[1];
                            const paddedHeight = newHeight + 20;
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
