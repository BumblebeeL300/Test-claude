// One place for every "what does this app do?" string. Consumed by the spec plate, rail tooltips,
// Switchboard modules, the ⌘K palette and Settings. Plain language, no jargon.
// `roles` are NIM model roles (see nimlab.core.registry.Role) shown as small chips.
export const DESCRIPTIONS = {
  foundry: {
    name: 'Foundry',
    line: 'Describe a piece of software in plain words and an AI coder plans it, writes it with tests, and fixes it until the tests pass.',
    short: 'Describe software; it plans, builds and tests it',
    steps: ['Describe what to build', 'Pick a project folder', 'Watch it build and read the report'],
    roles: ['REASONING', 'CODER', 'CHAT'],
  },
  council: {
    name: 'Council',
    line: 'Ask one question and let several AI models work on it together, then get one answer plus how much they agreed.',
    short: 'Several AI models, one combined answer',
    steps: ['Type your question', 'Pick a way for them to team up', 'Read the verdict'],
    roles: ['CHAT', 'REASONING', 'REWARD', 'EMBED'],
  },
  atlas: {
    name: 'Atlas',
    line: 'Ask questions about your own files and get answers that point to the exact passages they came from.',
    short: 'Ask your own documents, with sources',
    steps: ['Add your documents', 'Ask a question', 'Check the numbered sources'],
    roles: ['EMBED', 'RERANK', 'CHAT', 'SAFETY'],
  },
  arena: {
    name: 'Arena',
    line: 'Pit AI models against each other on test questions to see which is best, and send each prompt to the right one.',
    short: 'Compare models and pick the best for each prompt',
    steps: ['Pick models and a test set', 'Run the match', 'Read the leaderboard or try the router'],
    roles: ['CHAT', 'EMBED'],
  },
  prism: {
    name: 'Prism',
    line: 'Give it a picture and it describes it, reads the text, pulls out data, or turns a screenshot into a web page.',
    short: 'Understand images: describe, read, extract, rebuild',
    steps: ['Drop an image', 'Pick a mode', 'Read the print'],
    roles: ['VISION'],
  },
  sentinel: {
    name: 'Sentinel',
    line: 'Safely test how well AI models resist tricks like prompt injection and leaks, and get a grade from A to F.',
    short: 'Test how well models resist misuse',
    steps: ['Pick models to test', 'Run the scan', 'Read the report card'],
    roles: ['CHAT', 'SAFETY'],
  },
  loom: {
    name: 'Loom',
    line: 'Build your own multi-step AI workflow by dragging blocks onto a canvas and connecting them with cables.',
    short: 'Wire AI steps into your own workflow',
    steps: ['Add blocks from the palette', 'Connect them with cables', 'Press run'],
    roles: ['CHAT', 'REASONING', 'VISION', 'EMBED'],
  },
  distill: {
    name: 'Distill',
    line: 'Make a clean set of example data for training AI: it writes, scores, filters and de-duplicates the examples.',
    short: 'Create clean training data for AI',
    steps: ['Choose a recipe and size', 'Start the run', 'Download the dataset'],
    roles: ['CHAT', 'REASONING', 'REWARD', 'EMBED'],
  },
};
export const describe = (id) => DESCRIPTIONS[id === 'forge' ? 'foundry' : id] || null;
