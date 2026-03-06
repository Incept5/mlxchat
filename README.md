# MLXChat

`MLXChat` is an iPhone and iPad SwiftUI app for running local Qwen3.5 MLX models with optional tool use and image input.

The current product target is a practical on-device chat app for modern Apple hardware, with `Qwen3.5-4B-MLX-4bit` as the default recommendation and larger local MLX variants available up to `Qwen3.5-9B-MLX-4bit`.

## Product Direction

- iPhone-first chat UX with iPad support
- local inference with MLX
- conservative tool use for current-info and URL tasks
- text-first performance, with image handling available when needed
- modern system UI that stays close to Apple defaults so newer iOS releases inherit the latest material and glass treatments naturally

## Highlights

- local Qwen3.5 chat with streaming responses
- optional reasoning mode
- optional web search via Brave Search API
- direct URL fetch support
- image input support for compatible models
- downloadable model picker with:
  - progress ring
  - percentage
  - cancel button
  - automatic retry on transient download interruption
- Markdown rendering in assistant bubbles
- configurable system prompt with runtime variables such as date, time, locale, and approximate location
- macOS `ToolTest` harness for prompt and tool iteration outside the phone UI

## Repository Layout

- `MLXChat/` iOS app source
- `MLXChat.xcodeproj/` checked-in Xcode project
- `project.yml` XcodeGen spec
- `ToolTest/` macOS command-line test harness for model and tool behavior

## Requirements

- Xcode 16 or newer
- Apple Silicon recommended
- iOS 26 deployment target
- local disk space for MLX models

## Models

This repository does not ship model weights.

The app is set up around Hugging Face repos under the `mlx-community` namespace. The current picker is intentionally limited to Qwen3.5 MLX variants:

- `Qwen3.5-0.8B-MLX-4bit`
- `Qwen3.5-0.8B-MLX-8bit`
- `Qwen3.5-0.8B-MLX-bf16`
- `Qwen3.5-2B-MLX-4bit`
- `Qwen3.5-2B-MLX-8bit`
- `Qwen3.5-2B-MLX-bf16`
- `Qwen3.5-4B-MLX-4bit`
- `Qwen3.5-4B-MLX-8bit`
- `Qwen3.5-4B-MLX-bf16`
- `Qwen3.5-9B-MLX-4bit`

Recommended default:

- `mlx-community/Qwen3.5-4B-MLX-4bit`

Why this default:

- materially better quality than the tiny models
- still usable on-device
- better latency than the larger `9B` variant on iPhone

## Qwen3.5 Notes

This app is tuned specifically around Qwen3.5 behavior rather than older Qwen families.

Important practical points:

- Qwen3.5 should be controlled with `enable_thinking`, not prompt hacks like `/think` or `/no_think`
- text-only turns should prefer the LLM path for speed
- tools work better when obvious current-info and URL tasks are prefetched by the app instead of relying purely on model-generated tool markup
- prompt and template details matter a lot for visible reasoning leakage

The codebase contains specific safeguards for:

- disabling visible reasoning where possible
- cutting off repetition loops
- handling tool calls conservatively
- keeping normal chat on the faster text-only path

## Tools

Supported tools:

- `web_search`
- `url_fetch`

Notes:

- `web_search` requires a Brave Search API key
- `url_fetch` works without Brave
- the app prefetches obvious URL and current-info requests before generation, then asks the model to answer from the retrieved result

## Tool-Calling Tips For Other Technologists

If you are trying to build a similar local-agent experience with Qwen3.5 and MLX, the main lessons from this app are:

1. Prefer app-driven tool prefetch for obvious tasks.
2. Do not assume prompt text alone will reliably disable reasoning on Qwen3.5.
3. Keep tool schemas out of ordinary chat prompts unless tools are actually needed.
4. Treat text chat and image chat as different runtime paths.
5. Measure model generation separately from preprocessing and tool time.
6. Expect cancellation and early-stop handling to matter for UI responsiveness.

Concrete examples:

- a direct URL prompt should run `url_fetch` first
- a recent-news prompt should trigger search before model generation when a search key is configured
- a plain `Hello` should not carry tool schemas or spend time on reasoning

## Apple Platform Notes

The UI intentionally stays close to system components:

- `NavigationStack`
- `TabView`
- `List`
- standard toolbars
- system materials for persistent bars and insets

That is deliberate. On current iOS releases, including the newer glass-oriented visual system, standard containers and materials age better than custom chrome and require less maintenance.

Practical guidance followed here:

- use system navigation and tab patterns
- keep controls legible over material backgrounds
- avoid custom ornamental surfaces unless they materially improve clarity
- let the system handle most of the visual treatment

## Running The App

1. Open `MLXChat.xcodeproj` in Xcode.
2. Select a device or simulator that supports the `iOS 26` deployment target.
3. Build and run the `MLXChat` scheme.
4. Open `Settings`.
5. Set:
   - `Model Download Limit` if you want to constrain the picker
   - `GPU Memory Limit` for runtime memory use
   - `Brave Search API Key` if you want web search
6. Open `Select Model` and download a model.

## Download UX

Model downloads now start directly from the model picker.

While a model is downloading:

- the picker row shows a circular progress indicator
- the picker row shows a percentage
- the picker row exposes cancel
- Settings shows the same active download state
- interrupted downloads retry once automatically

## Settings

Key settings:

- `GPU Memory Limit`
- `Model Download Limit`
- `Context Size`
- `Stream Responses`
- image compression controls
- system prompt
- tools enablement and Brave API key

System prompt variables:

- `{today}`
- `{date}`
- `{time}`
- `{datetime}`
- `{timestamp}`
- `{unixtime}`
- `{location}`
- `{address}`
- `{coordinates}`
- `{latitude}`
- `{longitude}`
- `{timezone}`
- `{locale}`
- `{device}`
- `{system}`
- `{version}`
- `{username}`

## ToolTest

`ToolTest` is kept because it is useful for faster iteration on prompt, tool, and model behavior without repeatedly deploying to a device.

Build and run:

```bash
cd ToolTest
swift build
swift run
```

## Project Generation

The checked-in Xcode project is intended to be usable directly.

If you prefer regenerating it:

```bash
xcodegen generate
```

## Known Constraints

- Qwen3.5 still needs defensive handling around reasoning leakage
- larger models are practical on iPad before iPhone
- image turns are slower than text-only turns
- Brave-backed search quality depends on the configured API key

## Publishing Checklist

Before publishing or sharing builds:

1. confirm the app icon and bundle identifier are correct
2. confirm location, camera, and photo permission strings are intentional
3. verify the selected deployment target is what you want to support
4. test one plain chat, one image turn, and one tool-backed turn on a real device
5. remove any private API keys from local settings before screenshots or demos

## GitHub

This repository is prepared to be committed as a normal Xcode project with the app, assets, and `ToolTest` harness checked in.

Typical flow:

```bash
git add -A
git commit -m "Initial commit for MLXChat"
git remote add origin <your-repo-url>
git push -u origin main
```
