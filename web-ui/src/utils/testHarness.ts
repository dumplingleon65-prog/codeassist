// Builds a Python test harness that reads a single line from stdin containing
// comma-separated "name = value" pairs, reconstructs kwargs, calls the given
// entry point, and prints the result. Mirrors the harness used by the state-service.
export function buildTestHarness(entryPoint: string, userCode: string): string {
  if (!entryPoint || typeof entryPoint !== "string") {
    throw new Error("entryPoint must be a non-empty string");
  }

  const HARNESS = String.raw`
import sys, ast

# Split a comma-separated assignment list into top-level segments, respecting
# brackets/braces/parentheses and quoted strings so inner commas don't split.
# Example: "nums = [1,2,3], target = 9" -> ["nums = [1,2,3]", "target = 9"]
def split_top_level(s):
    parts, buf = [], []
    depth, in_str, quote, esc = 0, False, None, False
    for ch in s:
        if in_str:
            buf.append(ch)
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == quote:
                in_str, quote = False, None
            continue
        # Enter string mode when we see a quote at top level
        if ch in ('"', "'"):
            in_str, quote = True, ch
            buf.append(ch)
            continue
        # Track nesting depth for lists/tuples/dicts
        if ch in '([{':
            depth += 1
            buf.append(ch)
            continue
        if ch in ')]}':
            depth -= 1
            buf.append(ch)
            continue
        # Only split on commas at depth 0 and outside strings
        if ch == ',' and depth == 0:
            parts.append(''.join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    if buf:
        parts.append(''.join(buf).strip())
    return [p for p in parts if p]

# Parse "name = value" pairs into kwargs using safe literal_eval for RHS.
# Unsupported fragments are ignored (no '='), consistent with dataset format.
def parse_assignments(s):
    kwargs = {}
    if not s:
        return kwargs
    for part in split_top_level(s):
        if '=' not in part:
            continue
        name, val = part.split('=', 1)
        name, val = name.strip(), val.strip()
        kwargs[name] = ast.literal_eval(val)
    return kwargs

# Entrypoint: read stdin once, build kwargs, call the dataset's entry_point, print.
# An empty or "none" input means no arguments (some checks are zero-arg).
def __run_entry():
    input_line = sys.stdin.read().strip()
    if input_line.lower() in ('', 'none'):
        kwargs = {}
    else:
        kwargs = parse_assignments(input_line)
    func = ENTRY_POINT
    res = func(**kwargs)
    # Print delimiter to separate user stdout from harness output
    print('---HARNESS_OUTPUT---')
    print(res)

__run_entry()
`;

  // Bind ENTRY_POINT symbol
  const harness = HARNESS.replace("ENTRY_POINT", entryPoint);
  return `${userCode}\n\n${harness}\n`;
}

