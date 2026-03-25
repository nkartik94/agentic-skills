---
name: production-react
description: "Production-grade React coding conventions for components, hooks, TypeScript, state management, data fetching, testing, and accessibility. Use this skill when writing React components, creating custom hooks, setting up project structure, managing state, writing tests, or reviewing React code for production readiness. Triggers on any task involving .tsx/.jsx files, React hooks, component libraries, or frontend architecture."
license: MIT
metadata:
  author: nkartik94
  version: "1.0.0"
---

# Production React

Apply every rule below whenever writing or reviewing React code.

## When to Apply

- Writing new React components, hooks, or pages
- Setting up project structure or feature modules
- Managing state (local, global, or server)
- Writing tests for React components or hooks
- Reviewing React code for production readiness
- Configuring TypeScript, ESLint, or Vite for a React project

---

## Quick Reference

| Rule | Pattern |
|------|---------|
| Project structure | Feature-based: `src/features/<name>/{components,hooks,api,types}/` |
| Component files | `PascalCase.tsx`, one component per file |
| Hooks | `useXxx.ts`, must start with `use` + capital |
| Props | `interface XxxProps {}` in same file as component |
| Boolean props | `is`, `has`, `can`, `should` prefixes |
| Event callbacks | `on` prefix — `onClick`, `onSubmit`, `onClose` |
| Imports | Absolute `@/` alias, never relative `../../../` |
| Barrel files | Avoid — direct file imports for tree-shaking |
| Local state | `useState` simple, `useReducer` complex/related |
| Global state | Zustand — never Context for frequently-changing state |
| Data fetching | TanStack Query (`useQuery`/`useMutation`) only |
| Forms | React Hook Form + Zod schema validation |
| Testing | React Testing Library — `getByRole`/`getByLabelText` |
| Lazy loading | `React.lazy` + `Suspense` at route boundaries |
| Accessibility | Semantic HTML first, ARIA only when HTML is insufficient |

---

## 1. Project Structure

Feature-based architecture — each feature is self-contained:

```
src/
├── app/
│   ├── router.tsx          # Route definitions
│   ├── providers.tsx       # Provider tree (QueryClient, ThemeProvider, etc.)
│   └── main.tsx            # Entry point
├── features/               # PRIMARY organization unit
│   └── <feature-name>/
│       ├── api/            # Feature-specific queries & mutations
│       ├── components/     # Feature-scoped components
│       ├── hooks/          # Feature-scoped custom hooks
│       ├── types/          # Feature TypeScript types
│       └── utils/          # Feature utilities
├── components/
│   └── ui/                 # Shared atomic components (Button, Modal, Badge)
├── hooks/                  # Shared custom hooks (used by 2+ features)
├── lib/                    # Pre-configured third-party instances
│   ├── queryClient.ts      # TanStack Query client
│   └── axios.ts            # Axios instance with interceptors
├── stores/                 # Global Zustand stores
├── types/                  # Shared TypeScript types
├── utils/                  # Shared utilities
├── config/
│   └── config.ts           # Typed environment variables
└── testing/
    ├── test-utils.tsx      # Custom render with providers
    └── mocks/              # Shared mock data
```

**Import flow rule:** `shared components/hooks/utils` → `features` → `app`. Cross-feature imports are forbidden — if two features need the same thing, move it to `src/shared/` or `src/components/`.

Start colocated — put everything in the feature folder. Move to a shared location only when used by 2+ features.

---

## 2. File & Component Naming

```
components/
  UserCard.tsx          ✅ PascalCase for components
  user-card.test.tsx    ✅ lowercase-kebab for test files
  userCard.utils.ts     ✅ camelCase for non-component TS files
hooks/
  useUserData.ts        ✅ useXxx — hooks only
  formatDate.ts         ✅ camelCase — regular utilities
```

Rules:
- One component per file — no exceptions for exported components
- Multiple small, non-exported sub-components are allowed in the same file
- Test files live next to the file they test: `Button.tsx` → `Button.test.tsx`
- No `index.tsx` for individual components — import by filename

---

## 3. TypeScript Props

```tsx
// ✅ Interface co-located with component, exported when reused
interface UserCardProps {
  /** User's full name */
  name: string;
  email?: string;
  isActive?: boolean;
  /** Called when the card is clicked */
  onClick?: (userId: string) => void;
  children?: React.ReactNode;
}

function UserCard({ name, email, isActive = false, onClick, children }: UserCardProps) {
  return <div onClick={() => onClick?.(name)}>{name}</div>;
}
```

**Key types:**

```tsx
// JSX content
children: React.ReactNode           // Any valid JSX (text, elements, arrays, null)

// Event handlers
onChange: React.ChangeEventHandler<HTMLInputElement>
onSubmit: (event: React.FormEvent<HTMLFormElement>) => void
onClick: React.MouseEventHandler<HTMLButtonElement>

// Style
style?: React.CSSProperties

// Extend native HTML props
interface CardProps extends React.ComponentPropsWithoutRef<'div'> {
  title: string;
  // Card now accepts all <div> props (id, className, aria-*, etc.)
}
```

Rules:
- Always destructure props in function params — never `props.xxx`
- Use `interface` for component props; `type` for unions and computed types
- Provide default values in destructuring, not in the body
- Never use `any` — prefer `unknown` or proper generics

---

## 4. JSX Conventions

```tsx
// ✅ Double quotes for JSX attributes
<input className="form-input" placeholder="Enter email" />

// ✅ Self-closing tags — no space before />
<Button isLoading />
<img src={avatar} alt="User avatar" />

// ✅ No spaces inside JSX curly braces
<div>{value}</div>           // ✅
<div>{ value }</div>         // ❌

// ✅ Boolean prop shorthand (omit ={true})
<Button isLoading />         // ✅
<Button isLoading={true} />  // ❌ verbose

// ✅ Conditional rendering
{isVisible && <Modal />}
{isLoading ? <Spinner /> : <Content />}

// ✅ No inline styles
<div style={{ color: 'red' }} />      // ❌
<div className={styles.error} />      // ✅

// ✅ Wrap multi-line JSX in parentheses
return (
  <div className="container">
    <Header />
    <Main />
  </div>
);
```

---

## 5. Custom Hooks

```tsx
// ✅ Naming: use + PascalCase, describes what it does (not when)
export function useOnlineStatus() { }    // ✅ specific behavior
export function useChatRoom() { }        // ✅ concrete use case
function useEffectOnce() { }             // ❌ lifecycle wrapper (anti-pattern)

// ✅ One concern per hook — extract if it does two things
export function useUserProfile(userId: string) {
  const { data: user, isLoading, error } = useQuery({
    queryKey: ['user', userId],
    queryFn: () => fetchUser(userId),
  });

  return { user, isLoading, error };
}

// ✅ Always return a consistent shape
export function useToggle(initial = false) {
  const [value, setValue] = useState(initial);
  const toggle = useCallback(() => setValue(v => !v), []);
  return { value, toggle, setTrue: () => setValue(true), setFalse: () => setValue(false) };
}
```

Rules:
- Custom hooks MUST start with `use` followed by a capital letter
- Never call hooks conditionally or in loops
- A function that doesn't call other hooks is not a hook — don't prefix with `use`
- Keep hooks focused — if it does two unrelated things, split it

---

## 6. State Management

### Local and component state

```tsx
// ✅ useState for simple, independent values
const [isOpen, setIsOpen] = useState(false);

// ✅ useReducer for complex state with related updates
type Status = 'idle' | 'loading' | 'success' | 'error';
type State =
  | { status: 'idle' | 'loading' }
  | { status: 'success'; data: User }
  | { status: 'error'; error: string };

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case 'FETCH_START': return { status: 'loading' };
    case 'FETCH_SUCCESS': return { status: 'success', data: action.payload };
    case 'FETCH_ERROR': return { status: 'error', error: action.message };
    default: return state;
  }
}
```

### Global state — Zustand

```tsx
import { create } from 'zustand';

interface AppStore {
  theme: 'light' | 'dark';
  setTheme: (theme: 'light' | 'dark') => void;
}

export const useAppStore = create<AppStore>((set) => ({
  theme: 'light',
  setTheme: (theme) => set({ theme }),
}));

// ✅ Use selectors to avoid unnecessary re-renders
const theme = useAppStore((state) => state.theme);     // ✅ Only re-renders when theme changes
const store = useAppStore();                             // ❌ Re-renders on any store change
```

Rules:
- Context API is for static/slow-changing data (theme, locale, auth user)
- Never use Context for frequently-changing state — use Zustand
- One Zustand store per domain concern, not one giant store
- Full Zustand template with devtools and persist: [references/REFERENCE.md](references/REFERENCE.md)

---

## 7. Data Fetching

Use TanStack Query for all server state. Never fetch data inside raw `useEffect`.

```tsx
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';

// ✅ Query key convention: [entity, id/filters] tuple
const { data: user, isLoading, error } = useQuery({
  queryKey: ['users', userId],          // Unique, descriptive
  queryFn: () => fetchUser(userId),
  staleTime: 5 * 60 * 1000,            // 5 minutes
});

// ✅ Mutation with cache invalidation
const queryClient = useQueryClient();
const { mutate: updateUser, isPending } = useMutation({
  mutationFn: (data: UserUpdate) => api.patch(`/users/${userId}`, data),
  onSuccess: () => {
    queryClient.invalidateQueries({ queryKey: ['users', userId] });
  },
});
```

Rules:
- Always handle `isLoading`, `isError`, and empty-data states in UI
- Use `staleTime` to avoid unnecessary refetches
- Use `queryKey` arrays — never strings — for reliable cache invalidation
- Parallel requests via `Promise.all` in `queryFn`, not stacked `enabled` flags
- Full patterns (optimistic updates, prefetching, pagination): [references/REFERENCE.md](references/REFERENCE.md)

---

## 8. Forms

```tsx
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';

const schema = z.object({
  email: z.string().email('Invalid email address'),
  password: z.string().min(8, 'Minimum 8 characters'),
});

type FormData = z.infer<typeof schema>;

function LoginForm({ onSubmit }: { onSubmit: (data: FormData) => void }) {
  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<FormData>({
    resolver: zodResolver(schema),
    mode: 'onBlur',                     // Validate on blur, not on every keystroke
  });

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <label htmlFor="email">Email</label>
      <input id="email" {...register('email')} />
      {errors.email && <span role="alert">{errors.email.message}</span>}

      <button type="submit" disabled={isSubmitting}>
        {isSubmitting ? 'Signing in...' : 'Sign in'}
      </button>
    </form>
  );
}
```

Rules:
- Always define schema with Zod — validation logic lives in the schema, not the component
- Use uncontrolled mode (`register`) by default; use `Controller` only for external UI library inputs
- Validate on `onBlur`, not `onChange` (avoids error flicker while typing)
- Never destructure the entire `methods` object — extract only what you use

---

## 9. Error Handling

```tsx
// ✅ Granular error boundaries — wrap sections, not the whole page
function ProductPage() {
  return (
    <ErrorBoundary FallbackComponent={PageFallback}>
      <Header />
      <ErrorBoundary FallbackComponent={SectionFallback}>
        <ProductDetails />
      </ErrorBoundary>
      <ErrorBoundary FallbackComponent={SectionFallback}>
        <Reviews />
      </ErrorBoundary>
    </ErrorBoundary>
  );
}

// ✅ Async errors (event handlers) — error boundaries don't catch these
async function handleSubmit() {
  try {
    await api.post('/orders', data);
    setSuccess(true);
  } catch (error) {
    setErrorMessage('Failed to submit. Please try again.');
  }
}

// ✅ Never expose raw error messages in production UI
<div>{error.message}</div>          // ❌ leaks internals
<div>Something went wrong.</div>    // ✅
```

Rules:
- Always use `react-error-boundary` package (not hand-rolled class components)
- Log errors to Sentry/error tracking inside `FallbackComponent`
- Error boundaries catch render errors only — use try/catch for async event handlers
- TanStack Query handles async data errors — handle `isError` state in UI

---

## 10. Performance

**Measure first, optimize second** — use React DevTools Profiler before adding any memoization.

```tsx
// ✅ React.lazy + Suspense at route level (always do this)
const Dashboard = lazy(() => import('./routes/Dashboard'));
const Settings = lazy(() => import('./routes/Settings'));

<Suspense fallback={<PageSpinner />}>
  <Routes>
    <Route path="/dashboard" element={<Dashboard />} />
    <Route path="/settings" element={<Settings />} />
  </Routes>
</Suspense>

// ✅ memo + useCallback only when profiling shows a bottleneck
const ExpensiveList = memo(function ExpensiveList({ items }: Props) {
  return <ul>{items.map(item => <Item key={item.id} item={item} />)}</ul>;
});

const handleClick = useCallback(() => doSomething(id), [id]);

// ✅ Architectural fix first — move state down before memoizing
// Instead of memoizing, extract stateful logic into a child component
```

Anti-patterns:
- Never use array index as `key` — use stable unique IDs
- Never create objects or functions inline in JSX props passed to memoized components
- Never add `useMemo`/`useCallback` pre-emptively — measure first
- Avoid `useEffect` for derived state — compute inline or use `useMemo`

---

## 11. Accessibility

```tsx
// ✅ Semantic HTML is always first choice
<button onClick={handleClick}>Submit</button>    // ✅
<div onClick={handleClick}>Submit</div>          // ❌ not accessible

// ✅ Every form input has a paired label
<label htmlFor="email">Email address</label>
<input id="email" type="email" required />

// ✅ Images — descriptive alt for meaningful, empty for decorative
<img src="profile.jpg" alt="Jane Doe's profile picture" />
<img src="decorative-divider.svg" alt="" />

// ✅ Focus management in modals
function Dialog({ onClose }: { onClose: () => void }) {
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    closeRef.current?.focus();
  }, []);

  return (
    <div role="dialog" aria-modal="true" aria-labelledby="dialog-title">
      <h2 id="dialog-title">Confirm action</h2>
      <button ref={closeRef} onClick={onClose}>Close</button>
    </div>
  );
}
```

Rules:
- Use ARIA attributes only when semantic HTML is insufficient
- All interactive elements must be reachable and operable with keyboard
- Color must not be the only means of conveying information
- Run `eslint-plugin-jsx-a11y` in CI — fix all a11y lint errors

---

## 12. Testing

```tsx
import { render, screen } from '@/testing/test-utils'; // custom render with providers
import userEvent from '@testing-library/user-event';

// ✅ Arrange-Act-Assert
test('submits form with valid input', async () => {
  const user = userEvent.setup();
  const onSubmit = vi.fn();

  // Arrange
  render(<LoginForm onSubmit={onSubmit} />);

  // Act
  await user.type(screen.getByLabelText(/email/i), 'user@example.com');
  await user.type(screen.getByLabelText(/password/i), 'secret123');
  await user.click(screen.getByRole('button', { name: /sign in/i }));

  // Assert
  expect(onSubmit).toHaveBeenCalledWith({
    email: 'user@example.com',
    password: 'secret123',
  });
});

// ✅ Test error states
test('shows validation error when email is blank', async () => {
  render(<LoginForm onSubmit={vi.fn()} />);
  await userEvent.click(screen.getByRole('button', { name: /sign in/i }));
  expect(screen.getByRole('alert')).toHaveTextContent(/invalid email/i);
});
```

Query priority (highest to lowest): `getByRole` > `getByLabelText` > `getByPlaceholderText` > `getByText` > `getByTestId`

Rules:
- Never query by CSS class or DOM structure — test what users see
- Never test internal state, refs, or implementation details
- Mock at module boundary (`vi.mock('...')`), not at the component level
- Always wrap in custom `render` from `testing/test-utils.tsx` (includes all providers)
- Full testing utilities and hook testing examples: [references/REFERENCE.md](references/REFERENCE.md)

---

## 13. Import Conventions

```tsx
// ✅ Import groups (in order, blank line between each)
import { useState, useCallback } from 'react';               // 1. React
import { useQuery } from '@tanstack/react-query';            // 2. Third-party
import { z } from 'zod';

import { Button } from '@/components/ui/Button';             // 3. Internal shared
import { useAuth } from '@/hooks/useAuth';

import { UserCard } from './UserCard';                        // 4. Relative (same feature)
import type { UserCardProps } from './UserCard.types';        // 5. Type imports last

// ✅ Absolute imports via @/ alias — never deep relative paths
import { formatDate } from '@/utils/date';                   // ✅
import { formatDate } from '../../../../utils/date';          // ❌

// ✅ Direct imports — not barrel files
import { Button } from '@/components/ui/Button';             // ✅
import { Button } from '@/components';                        // ❌ loads everything
```

---

## 14. Styling

Choose **one** approach per project and apply it consistently:

```tsx
// Option A: CSS Modules (scoped, TypeScript-friendly)
import styles from './Button.module.css';

function Button({ variant }: Props) {
  return (
    <button className={`${styles.button} ${styles[variant]}`}>
      Click
    </button>
  );
}

// Option B: Tailwind CSS (utility-first)
function Button({ isLoading }: Props) {
  return (
    <button className={`px-4 py-2 rounded font-medium bg-blue-600 text-white
      ${isLoading ? 'opacity-50 cursor-not-allowed' : 'hover:bg-blue-700'}`}>
      Click
    </button>
  );
}
```

Rules:
- Never use inline `style` prop — use CSS classes
- Use CSS custom properties (`var(--color-primary)`) for theme values
- With Tailwind: use `clsx` or `cva` for conditional class logic — no template literal chains

---

## 15. Component Composition

```tsx
// ✅ Accept children for flexible layouts
function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className={styles.card}>
      <h3>{title}</h3>
      {children}
    </div>
  );
}

// ✅ Compound components for related pieces
function Tabs({ children }: { children: React.ReactNode }) {
  const [active, setActive] = useState(0);
  return <TabsContext.Provider value={{ active, setActive }}>{children}</TabsContext.Provider>;
}
Tabs.Panel = TabPanel;
Tabs.Trigger = TabTrigger;

// Usage: <Tabs><Tabs.Trigger /><Tabs.Panel /></Tabs>
```

Rules:
- Avoid prop drilling beyond 2 levels — use Context or state management
- Prefer composition (`children`, slots) over configuration props
- Keep components focused — if it needs >5–7 props, consider splitting

---

## 16. Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Component | PascalCase | `UserCard`, `ProductList` |
| Component file | PascalCase + `.tsx` | `UserCard.tsx` |
| Hook | `useXxx` | `useUserProfile`, `useToggle` |
| Hook file | camelCase + `.ts` | `useUserProfile.ts` |
| Utility | camelCase | `formatDate`, `validateEmail` |
| Constant | `UPPER_SNAKE_CASE` | `MAX_RETRY_COUNT`, `API_BASE_URL` |
| Props interface | `XxxProps` | `UserCardProps`, `ButtonProps` |
| Type | PascalCase | `UserStatus`, `ApiResponse<T>` |
| Boolean prop | `is/has/can/should` prefix | `isLoading`, `hasError`, `canEdit` |
| Event handler | `on` prefix | `onClick`, `onSubmit`, `onClose` |
| Test file | same name + `.test.tsx` | `UserCard.test.tsx` |
| CSS module | same name + `.module.css` | `UserCard.module.css` |

---

## 17. Anti-Patterns

```tsx
// ❌ Array index as key — breaks reconciliation when order changes
items.map((item, index) => <Item key={index} />)
// ✅ items.map(item => <Item key={item.id} />)

// ❌ Mutating state directly
state.items.push(newItem)
// ✅ setState([...state.items, newItem])

// ❌ Fetching in useEffect
useEffect(() => { fetch('/api/users').then(setUsers); }, []);
// ✅ Use TanStack Query

// ❌ Derived state in useState
const [fullName, setFullName] = useState(`${first} ${last}`);
// ✅ const fullName = `${first} ${last}`  — compute inline

// ❌ Giant single Context that causes all consumers to re-render
const AppContext = createContext({ user, theme, cart, settings });
// ✅ Split into UserContext, ThemeContext, CartContext

// ❌ Anonymous components (breaks React DevTools, errors)
export default () => <div>Hello</div>
// ✅ export function MyPage() { return <div>Hello</div> }

// ❌ useEffect for event listeners without cleanup
useEffect(() => { window.addEventListener('resize', handler); });
// ✅ return () => window.removeEventListener('resize', handler)

// ❌ Prop drilling through 3+ levels
<A><B><C onSubmit={fn} /></B></A>
// ✅ Use Context or state management
```

Also avoid: `any` in TypeScript, relative imports `../../../`, skipping error boundary wrapping on async-heavy pages, using `document.querySelector` in React code.

---

## 18. Pre-Commit Checklist

### Structure & Naming
- [ ] Feature code lives in `src/features/<name>/`; shared code in `src/components/` or `src/hooks/`
- [ ] Component files are `PascalCase.tsx`; one exported component per file
- [ ] Hook files start with `use` + capital letter
- [ ] Test file lives next to source: `Button.tsx` → `Button.test.tsx`

### TypeScript
- [ ] All props typed with `interface XxxProps` (no `any`)
- [ ] Props destructured in function params with defaults
- [ ] Event handler types use `React.MouseEventHandler<T>` etc., not `Function`
- [ ] `type` imports use `import type { ... }`

### JSX
- [ ] JSX attributes use double quotes
- [ ] No inline `style` props
- [ ] Boolean props omit `={true}`
- [ ] Self-closing tags for empty components

### Hooks & State
- [ ] No hook calls inside conditions or loops
- [ ] `useEffect` has correct dependency array; cleanup function where needed
- [ ] No derived state stored in `useState` — compute inline
- [ ] Global state uses Zustand, not Context (except for static data)
- [ ] No raw `fetch`/`axios` in `useEffect` — use TanStack Query

### Forms
- [ ] Zod schema defined for all forms
- [ ] `mode: 'onBlur'` in `useForm` config
- [ ] Every `input` has a paired `label` with matching `htmlFor`/`id`
- [ ] Errors displayed in an element with `role="alert"`

### Performance
- [ ] Routes use `React.lazy` + `Suspense`
- [ ] No array index used as `key`
- [ ] No new objects/functions created inline in props for memoized components

### Testing
- [ ] Tests query by `getByRole` or `getByLabelText`
- [ ] Tests use custom `render` from `testing/test-utils.tsx`
- [ ] Happy path, error state, and edge cases all tested

### Accessibility
- [ ] All interactive elements are `<button>` or `<a>` (not `<div>`)
- [ ] All images have `alt` text (empty string for decorative)
- [ ] `eslint-plugin-jsx-a11y` passes with no errors

---

## 19. Environment & Config

```ts
// src/config/config.ts — typed, validated env vars
const config = {
  apiBaseUrl: import.meta.env.VITE_API_BASE_URL as string,
  environment: import.meta.env.MODE,
  sentryDsn: import.meta.env.VITE_SENTRY_DSN as string,
} as const;

export default config;
```

Rules:
- All browser-accessible env vars must be prefixed `VITE_`
- Commit `.env.example` with all keys and descriptions; never commit `.env`
- Access env vars only through `config.ts` — never `import.meta.env.VITE_*` scattered throughout code
- Validate required vars at startup and throw if missing

---

## 20. Tooling

Required tools for every production React project:

| Tool | Purpose | Config file |
|------|---------|-------------|
| TypeScript | Type safety | `tsconfig.json` |
| Vite | Build & dev server | `vite.config.ts` |
| ESLint | Linting (react-hooks, jsx-a11y) | `eslint.config.js` |
| Prettier | Code formatting | `.prettierrc` |
| Vitest | Unit/integration tests | `vite.config.ts` |
| React Testing Library | Component tests | `src/testing/` |
| TanStack Query | Server state | `src/lib/queryClient.ts` |

Full config files (`vite.config.ts` with path aliases, `tsconfig.json`, ESLint flat config): [references/REFERENCE.md](references/REFERENCE.md).

---

For full templates (feature folder, complete component, custom hook, Zustand store, TanStack Query patterns, form template, error boundary, testing utilities, Vite/TypeScript config), see [references/REFERENCE.md](references/REFERENCE.md).
