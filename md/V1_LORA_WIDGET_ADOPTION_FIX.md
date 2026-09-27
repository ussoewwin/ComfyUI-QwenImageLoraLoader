# V1 LoRA Widget Identity Issue — Complete Technical Explanation (PR #54 Merge)

- Repository: `ussoewwin/ComfyUI-QwenImageLoraLoader`
- Local merge commit: `01826f8` (local `main`)
- Remote merge commit: `81e38b0` (`origin/main`, parents = `ff7e783` + `412b102`)
- PR content commit: `412b1028f3068373e60fe5635e194ca0ff5f5d1d` (single commit)
- Written: 2026-09-27
- Every fact below was measured (ComfyUI_frontend sources, the installed frontend package, repository sources, and executed verification harnesses). Nothing is speculative.

---

## ① What the problem was

### 1.1 Summary

The new ComfyUI frontend (ComfyUI_frontend) now "adopts" node widgets into concrete widget classes, **keeping object identity but replacing the widget's prototype**. The V1 node scripts decided "is this row mine?" with `w instanceof NunchakuLoraWidget`; after adoption that check is permanently `false`, so four interactions stopped working.

### 1.2 Mechanism (source by source)

1. When a node is constructed, the frontend calls `initializeWidgetsView(this)` (`src/lib/litegraph/src/LGraphNode.ts:1048`).
2. That redefines `node.widgets` as a getter returning `state.view`, where `state.view` is a **Proxy** produced by `createArrayMutationView(target, ...)` (`src/lib/litegraph/src/node/widgetsView.ts`, `src/lib/litegraph/src/infrastructure/createMutationView.ts`).
3. Array mutation methods (`copyWithin` / `fill` / `pop` / `push` / `reverse` / `shift` / `sort` / `splice` / `unshift`) or a whole-array assignment trigger a commit, which calls `syncWidgetOrder(node, widgets)`:

```ts
// src/lib/litegraph/src/node/widgetsView.ts (excerpt)
function syncWidgetOrder(node: LGraphNode, widgets: IBaseWidget[]): void {
  node._widgetSlotsDirty = true
  const graphId = node.graph?.rootGraph.id
  for (const [index, widget] of widgets.entries()) {
    const concreteWidget = toConcreteWidget(widget, node)   // wrapLegacyWidgets defaults to true
    widgets[index] = concreteWidget
    if (graphId) concreteWidget.setNodeId(node.id)
  }
  ...
}
```

4. `toConcreteWidget` instantiates the matching concrete class for known types; for `type === "custom"` (this extension's `NunchakuLoraWidget.type`) it falls through to the `default` branch and is wrapped as a `LegacyWidget` (`src/lib/litegraph/src/widgets/widgetMap.ts:205-280`).
5. The decisive part is `adoptConcreteWidget`: it does **not** swap in a new object, it adopts **in place** (same file, lines 139-175):

```ts
function adoptConcreteWidget<C extends BaseWidget>(widget: IBaseWidget, concrete: C): C {
  if (concrete === widget || !Object.isExtensible(widget)) return concrete
  const rawOptions = widget.options
  const descriptors = collectDescriptors(concrete)
  const foreignDescriptors = collectDescriptors(widget)
  for (const [key, foreignDescriptor] of foreignDescriptors) {
    if (key === 'options') continue
    const descriptor = mergeDescriptor(descriptors.get(key), foreignDescriptor,
      Object.getOwnPropertyDescriptor(widget, key))
    if (key === 'disabled' && foreignDescriptor.get) descriptor.get = foreignDescriptor.get
    descriptors.set(key, descriptor)
  }
  preserveHiddenFacade(descriptors, foreignDescriptors)
  if (Reflect.ownKeys(widget).some((key) => descriptors.has(key) &&
        Object.getOwnPropertyDescriptor(widget, key)?.configurable === false) ||
      !Reflect.setPrototypeOf(widget, Object.getPrototypeOf(concrete)))
    return concrete
  Object.defineProperties(widget, Object.fromEntries(descriptors))
  const adopted = widget as unknown as C
  if (adopted instanceof BaseWidget) adopted.options = rawOptions
  return adopted
}
```

In other words: **the same object**, whose prototype is replaced via `Reflect.setPrototypeOf`, and whose concrete-class members are installed as own properties. Therefore:

- references do not change: elements of `node.widgets` and any reference the extension holds still point at the same object;
- `Object.getPrototypeOf(widget)` is no longer `NunchakuLoraWidget.prototype`, so `w instanceof NunchakuLoraWidget === false`.

6. On any later commit, `instantiateConcreteWidget` returns the object as-is because `widget instanceof BaseWidget` (same file, lines 210-211), making `concrete === widget`; adoption then becomes idempotent and does not rewrite descriptors again.

### 1.3 The four broken check sites and their symptoms

`js/z_qwen_lora_dynamic_v1.js` / `js/zimageturbo_lora_dynamic_v1.js` (identical line numbers in both files):

| Line | Location | Purpose | Symptom once broken |
|---|---|---|---|
| 258 | `getSlotInPosition` | On right-click, decide whether the point is on a LoRA row | Row hit detection fails; the custom menu never opens |
| 272 | `getSlotMenuOptions` | Build the 4-item menu (toggle / move up / move down / remove) | Menu degrades to the default menu (no custom entries) |
| 309 | `moveLora` | Restrict movement to rows owned by this extension | Up/down movement is rejected |
| 322 | `onConfigure` | On workflow load, clear existing rows before rebuilding from `widgets_values` | Stale rows survive and rows are duplicated |

### 1.4 Frontend version boundary (measured)

| Frontend version | `widgetsView.ts` | Calls `toConcreteWidget` | `widgetMap.ts` has `adoptConcreteWidget` | V1 affected |
|---|---|---|---|---|
| 1.52.7 (installed here) | absent | — | no | no |
| v1.53.1 | 2165 bytes | no | no | no |
| v1.53.2 | 2165 bytes | no | no | no |
| **v1.53.3** | 2675 bytes | **yes** | **yes** (8381 bytes, includes `setPrototypeOf`) | **yes** |
| v1.53.6 (version pinned by ComfyUI master requirements) | 2675 bytes | yes | yes (9928 bytes) | yes |
| 1.55.x (frontend `3754918`, used by the PR regression) | 2675 bytes | yes | yes (10423 bytes) | yes |

- Distribution channels: the newest `comfyui-frontend-package` on PyPI is 1.54.7 (**no 1.55.x on PyPI**); the newest GitHub release of ComfyUI_frontend is **v1.55.11** (2026-09-20). `--front-end-version` pulls GitHub releases, the default pip install uses the pinned PyPI version.
- Same-root-cause report upstream: Comfy-Org/ComfyUI_frontend#17817 (rgthree Fast Groups Muter, frontend 1.55.9) is still open, i.e. upstream has not provided a general fix yet.

---

## ② PR overview

| Item | Content |
|---|---|
| PR | `ussoewwin/ComfyUI-QwenImageLoraLoader#54` |
| Title | Fix V1 LoRA widget interactions after frontend adoption |
| Author | DrJKL (Alexander Brown), from fork `DrJKL/ComfyUI-QwenImageLoraLoader`, branch `fix/widget-identity-v1` (`maintainer_can_modify = true`) |
| Commit | `412b1028` (single commit, subject `fix: preserve V1 LoRA widget behavior after adoption`; author `Amp <amp@ampcode.com>`; `Co-authored-by: Alexander Brown <drjkl@comfy.org>`) |
| Base | `ff7e783` (tip of `main` when the PR was opened) |
| Size | 2 files / +4 −4 each (8 lines total) |
| Timeline | opened 2026-09-22 → merged 2026-09-27 |
| Merge method | Remote: merge commit `81e38b0` (parents `ff7e783` + `412b102`, keeping the PR commit as second parent). Local: merge commit `01826f8` |
| Design decision | Do not test types with `instanceof`, and do not add persisted branding attributes; instead use **the node's existing live `loraWidgets` membership** as the identity contract |
| Author's regression test | On frontend `3754918`, loading the real extension modules through `initializeWidgetsView` / `createArrayMutationView` / `toConcreteWidget`. Before: both variants miss row hits and menus, reject a valid upward move, and restore one saved row alongside two stale rows. After: both variants hit the row, expose all four menu actions, reorder `[lora_1, lora_2]` to `[lora_2, lora_1]`, reject first-up and last-down boundary moves, and restore exactly `restored.safetensors` in both `widgets` and `loraWidgets`. `node --check` passes for both modules |
| Related upstream issue | https://github.com/Comfy-Org/ComfyUI_frontend/issues/17817 |

---

## ③ Files created / modified

### 3.1 Changed by this merge

| Kind | File | Change | Lines | Blob after merge |
|---|---|---|---|---|
| modified | `js/z_qwen_lora_dynamic_v1.js` | +4 −4 (4 sites) | 258 / 272 / 309 / 322 | `0f7cc5a91a667453ebe363446a0d84a0fc2f2dbb` |
| modified | `js/zimageturbo_lora_dynamic_v1.js` | +4 −4 (4 sites) | 258 / 272 / 309 / 322 | `bc0c563cc7b6cd2ed0b640711b8e3390ac47210a` |
| created | none | — | — | — |

The two files differ in only 3 lines (the module log line, the extension name, and the node-type guard). **Lines 250-341 — the region containing the four modified methods — are byte-for-byte identical** (`Compare-Object` reported no difference).

### 3.2 Deployment sync

Both files were copied to the runtime installation:

- `custom_nodes/ComfyUI-QwenImageLoraLoader/js/`
- Verified: SHA256 match (`z_qwen_lora_dynamic_v1.js` = `71790E5457DE98B3…`, `zimageturbo_lora_dynamic_v1.js` = `1592A39BEBA8C638…`)
- `git status` in that clone shows only these two files modified; nothing else changed

### 3.3 Scripts created for verification (reference material, not deliverables, not added to the repository)

- `harness/test.mjs` (V1, both files, 4 behaviours × 2 frontend contracts)
- `harness/test2.mjs` (V2/V3/V4 and widgethider, 3 contracts × 4 files)
- Both load the repository's real JS files unmodified and drive them.

---

## ④ Full code created / modified

### 4.1 Notes

This merge **creates no files and no functions**. It replaces four type checks with membership checks inside four existing methods (4 lines per file, 8 lines total). Therefore "the full code that was modified" below means: (a) the exact diff, and (b) the complete region containing the four modified methods (lines 250-341, identical in both files).

### 4.2 Exact diff

```diff
diff --git a/js/z_qwen_lora_dynamic_v1.js b/js/z_qwen_lora_dynamic_v1.js
index f8a7d0e..0f7cc5a 100644
--- a/js/z_qwen_lora_dynamic_v1.js
+++ b/js/z_qwen_lora_dynamic_v1.js
@@ -255,7 +255,7 @@ app.registerExtension({
 
       // If no slot was clicked, check if the click was on our LoraWidget
       const widget = this.widgets.find(w => {
-        return w instanceof NunchakuLoraWidget &&
+        return this.loraWidgets.includes(w) &&
           canvasY > (this.pos[1] + w.last_y) &&
           canvasY < (this.pos[1] + w.last_y + WIDGET_HEIGHT);
       });
@@ -269,7 +269,7 @@ app.registerExtension({
 
     // Show custom menu when the dummy slot is detected
     nodeType.prototype.getSlotMenuOptions = function (slot) {
-      if (slot && slot.widget instanceof NunchakuLoraWidget) {
+      if (slot && this.loraWidgets.includes(slot.widget)) {
         const widget = slot.widget;
         return [
           {
@@ -306,7 +306,7 @@ app.registerExtension({
       const idx = this.widgets.indexOf(widget);
       const targetIdx = idx + dir;
       // Ensure move range stays between LoraWidgets (prevent moving past or onto model/cpu_offload widgets)
-      if (this.widgets[targetIdx] instanceof NunchakuLoraWidget) {
+      if (this.loraWidgets.includes(this.widgets[targetIdx])) {
         this.widgets.splice(idx, 1);
         this.widgets.splice(targetIdx, 0, widget);
         this.setDirtyCanvas(true);
@@ -319,7 +319,7 @@ app.registerExtension({
       onConfigure?.apply(this, arguments);
       if (info.widgets_values) {
         // Clear initialized widgets
-        this.widgets = this.widgets.filter(w => !(w instanceof NunchakuLoraWidget));
+        this.widgets = this.widgets.filter(w => !this.loraWidgets.includes(w));
         this.loraWidgets = [];
 
         // Recover LoRA rows from serialized data
diff --git a/js/zimageturbo_lora_dynamic_v1.js b/js/zimageturbo_lora_dynamic_v1.js
index f0903b2..bc0c563 100644
--- a/js/zimageturbo_lora_dynamic_v1.js
+++ b/js/zimageturbo_lora_dynamic_v1.js
@@ -255,7 +255,7 @@ app.registerExtension({
 
       // If no slot was clicked, check if the click was on our LoraWidget
       const widget = this.widgets.find(w => {
-        return w instanceof NunchakuLoraWidget &&
+        return this.loraWidgets.includes(w) &&
           canvasY > (this.pos[1] + w.last_y) &&
           canvasY < (this.pos[1] + w.last_y + WIDGET_HEIGHT);
       });
@@ -269,7 +269,7 @@ app.registerExtension({
 
     // Show custom menu when the dummy slot is detected
     nodeType.prototype.getSlotMenuOptions = function (slot) {
-      if (slot && slot.widget instanceof NunchakuLoraWidget) {
+      if (slot && this.loraWidgets.includes(slot.widget)) {
         const widget = slot.widget;
         return [
           {
@@ -306,7 +306,7 @@ app.registerExtension({
       const idx = this.widgets.indexOf(widget);
       const targetIdx = idx + dir;
       // Ensure move range stays between LoraWidgets (prevent moving past or onto model/cpu_offload widgets)
-      if (this.widgets[targetIdx] instanceof NunchakuLoraWidget) {
+      if (this.loraWidgets.includes(this.widgets[targetIdx])) {
         this.widgets.splice(idx, 1);
         this.widgets.splice(targetIdx, 0, widget);
         this.setDirtyCanvas(true);
@@ -319,7 +319,7 @@ app.registerExtension({
       onConfigure?.apply(this, arguments);
       if (info.widgets_values) {
         // Clear initialized widgets
-        this.widgets = this.widgets.filter(w => !(w instanceof NunchakuLoraWidget));
+        this.widgets = this.widgets.filter(w => !this.loraWidgets.includes(w));
         this.loraWidgets = [];
 
         // Recover LoRA rows from serialized data
```

### 4.3 Full text of the four modified methods (identical in both files, lines 250-341)

```js
    // Right-click menu logic
    // Override getSlotInPosition to detect if a widget was clicked
    nodeType.prototype.getSlotInPosition = function (canvasX, canvasY) {
      const slot = LGraphNode.prototype.getSlotInPosition.apply(this, arguments);
      if (slot) return slot;

      // If no slot was clicked, check if the click was on our LoraWidget
      const widget = this.widgets.find(w => {
        return this.loraWidgets.includes(w) &&
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
      if (slot && this.loraWidgets.includes(slot.widget)) {
        const widget = slot.widget;
        return [
          {
            content: widget.value.enabled ? "? Toggle Off" : "?? Toggle On",
            callback: () => {
              widget.value.enabled = !widget.value.enabled;
              this.setDirtyCanvas(true);
            }
          },
          {
            content: "?? Move Up",
            callback: () => this.moveLora(widget, -1)
          },
          {
            content: "?? Move Down",
            callback: () => this.moveLora(widget, 1)
          },
          {
            content: "??? Remove",
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
      if (this.loraWidgets.includes(this.widgets[targetIdx])) {
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
        this.widgets = this.widgets.filter(w => !this.loraWidgets.includes(w));
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
```

> Note: the emoji inside the four menu labels display as `?` in some terminals. That is a terminal encoding artifact only; the file holds the original characters and this merge changed none of those strings.

---

## ⑤ What it means

### 5.1 Why `loraWidgets` is the correct identity contract

- Rows are created by `addLoraRow`: it does `new NunchakuLoraWidget(name, this)`, then `this.widgets.splice(btnIdx, 0, w)` and `this.loraWidgets.push(w)`. So `loraWidgets` holds **the same objects** that live in `widgets`.
- Adoption happens in place (`Reflect.setPrototypeOf` + `defineProperties`), so object identity is preserved and `loraWidgets.includes(x)` always agrees with an object-identity test.
- `instanceof`, by contrast, depends on frontend internals (whether the prototype is replaced). It is a **fragile dependency on an implementation detail** — which is precisely the shared root cause of the upstream report (#17817).

### 5.2 `this` binding equivalence (checked one by one)

| Site | How it is invoked | Result |
|---|---|---|
| `getSlotInPosition` | `LGraphCanvas.ts:8775` calls `node.getSlotInPosition(...)` | `this` = that node |
| `getSlotMenuOptions` | `LGraphCanvas.ts:8780` calls `node.getSlotMenuOptions(slot)` | `this` = that node |
| `moveLora` | Called from a menu callback as `this.moveLora(widget, ±1)` (arrow function inherits `this`) | `this` = that node |
| `onConfigure` | `onConfigure?.apply(this, arguments)` | `this` = that node |

### 5.3 Behaviour not touched by this change that is also unaffected by adoption (measured)

- **Custom `draw`**: neither `BaseWidget` nor `LegacyWidget` implements `draw`, and adoption keeps the original object's descriptors (including `draw`) as own properties. The draw entry point `LGraphNode.drawWidgets` (line 4200 onwards) sets `widget.last_y = y` (line 4218) and calls the custom `draw` (lines 4243-4244).
- **Custom `mouse` handling**: `LGraphCanvas.processWidgetClick` first tries `toConcreteWidget(widget, node, false)`; for an already-adopted object that returns `undefined`, so it falls through to `widget.mouse(...)` (line 3078) — the legacy path. Toggling, strength +/-, and LoRA selection keep working as before.
- **Saving**: `serialiseWidgetValues` (`LGraphNode.ts:184-196`) reads `widget.value`. After adoption `value` becomes `BaseWidget`'s accessor, but its `_state.value` still points at the original object, so the serialized output is unchanged (LoRA rows are still written into `widgets_values`).
- **Restoring**: `onConfigure` rebuilds rows from `info.widgets_values`; this change makes the "clear first" step actually work, which removes the duplicated-row failure.

### 5.4 Executed verification (real files loaded unmodified)

**a) V1, both files × 2 frontend contracts** (`test.mjs`)

| Code | Frontend contract | Row hit | Menu items | Move down | Rows after configure |
|---|---|---|---|---|---|
| before fix | 1.52.7 equivalent (no adoption) | hit | 4 | applied | 1 |
| before fix | new frontend equivalent (in-place adoption) | **miss** | **0** | **rejected** | **2 (stale)** |
| after fix | 1.52.7 equivalent | hit | 4 | applied | 1 |
| after fix | new frontend equivalent | hit | 4 | applied | 1 |

**b) V2 / V3 / V4 and widgethider.js × 3 contracts** (`test2.mjs`; contracts = 1.52.7 equivalent / new frontend equivalent / adoption after the extension hook is installed)

- Every file produced identical results under all three contracts: cached references still point at the objects inside `widgets`, writes through the cache reach the live widget, and re-adoption is idempotent.
- The `Object.defineProperty(w, 'value', …)` hook in `widgethider.js` keeps working both when adoption happens before the hook and when the hook is installed before adoption.

**c) Other checks**

- `node --check` on both files from the PR commit: exit = 0.
- Outside the two V1 files, no other file in the repository uses `NunchakuLoraWidget` type checks; V2/V3/V4 do not reference `NunchakuLoraWidget` or `loraWidgets` at all.

### 5.5 Immediate effect of merging in the current environment (1.52.7)

The installed frontend package is 1.52.7, and its bundle contains no `initializeWidgetsView` / `wrapLegacyWidgets` / `replaceNodeWidgetOrder` — there is no adoption mechanism. In that situation `loraWidgets` and the `NunchakuLoraWidget` type test select exactly the same set, so this change **does not alter current behaviour** (row 1 and row 3 of the table above are identical).

### 5.6 Remaining constraints and future risks

1. The fix relies on adoption preserving object identity. If `NunchakuLoraWidget` were ever given `Object.freeze`/`seal` or non-configurable properties, `adoptConcreteWidget` would fall back to returning a new object, and the `loraWidgets` membership test would **fail silently**.
2. `getSlotInPosition` is marked `@deprecated` on `LGraphNode` (replacement: `getSlotOnPos`). Current `main` still routes through the old one, but if that switches, this extension's right-click mechanism must be reimplemented.
3. The root fix belongs upstream (#17817). Until an official API exists, this PR's approach — using the node's own live array as the identity contract — is the correct available solution.

### 5.7 Final state of this merge

- Remote `main`: `81e38b0` (PR #54 MERGED, 2026-09-27T04:19:30Z)
- Local `main`: `01826f8` (content identical to `origin/main`; only the merge commit differs, each side ahead by one commit)
- Runtime: the two files under `custom_nodes/ComfyUI-QwenImageLoraLoader/js/` are synced with matching hashes; taking effect requires a ComfyUI restart or a browser reload
