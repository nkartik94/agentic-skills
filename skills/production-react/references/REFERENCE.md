# Production React — Reference

Extended templates, complete implementations, and configuration files. Load when you need the full pattern, not just the rule.

---

## Feature Folder Template

Full structure for a self-contained feature module:

```
src/features/users/
├── api/
│   ├── queries.ts          # useQuery hooks
│   └── mutations.ts        # useMutation hooks
├── components/
│   ├── UserCard.tsx
│   ├── UserCard.test.tsx
│   ├── UserCard.module.css
│   ├── UserList.tsx
│   └── UserList.test.tsx
├── hooks/
│   └── useUserPermissions.ts
├── types/
│   └── user.types.ts
├── utils/
│   └── formatUser.ts
└── index.ts                # Public API — only export what other features need
```

`index.ts` (feature public API):
```ts
export { UserCard } from './components/UserCard';
export { useUserPermissions } from './hooks/useUserPermissions';
export type { User, UserRole } from './types/user.types';
```

---

## Complete Component Template

```tsx
/**
 * UserCard
 *
 * Displays a user's profile summary with optional edit action.
 * Used in the Users list and the Dashboard recent-activity panel.
 */

import { memo, useCallback } from 'react';

import { Badge } from '@/components/ui/Badge';

import styles from './UserCard.module.css';

// --- Types ---

export interface UserCardProps {
  /** User's full display name */
  name: string;
  /** Email address shown as secondary label */
  email: string;
  /** Whether the user account is currently active */
  isActive?: boolean;
  /** Avatar image URL; falls back to initials if omitted */
  avatarUrl?: string;
  /** Called when the Edit button is clicked */
  onEdit?: (name: string) => void;
}

// --- Sub-components ---

function UserAvatar({ name, url }: { name: string; url?: string }) {
  if (url) return <img className={styles.avatar} src={url} alt={`${name}'s avatar`} />;
  return (
    <div className={styles.avatarFallback} aria-hidden="true">
      {name.slice(0, 2).toUpperCase()}
    </div>
  );
}

// --- Main component ---

export const UserCard = memo(function UserCard({
  name,
  email,
  isActive = false,
  avatarUrl,
  onEdit,
}: UserCardProps) {
  const handleEdit = useCallback(() => {
    onEdit?.(name);
  }, [name, onEdit]);

  return (
    <article className={styles.card}>
      <UserAvatar name={name} url={avatarUrl} />

      <div className={styles.info}>
        <h3 className={styles.name}>{name}</h3>
        <p className={styles.email}>{email}</p>
      </div>

      <Badge variant={isActive ? 'success' : 'neutral'}>
        {isActive ? 'Active' : 'Inactive'}
      </Badge>

      {onEdit && (
        <button className={styles.editButton} onClick={handleEdit} type="button">
          Edit
        </button>
      )}
    </article>
  );
});
```

---

## Custom Hook Template

```ts
// src/features/users/hooks/useUserPermissions.ts

import { useMemo } from 'react';

import { useAuthStore } from '@/stores/authStore';

import type { User, UserRole } from '../types/user.types';

// --- Types ---

interface UseUserPermissionsReturn {
  canEdit: boolean;
  canDelete: boolean;
  canViewAdmin: boolean;
}

// --- Hook ---

/**
 * Returns permission flags for the given user based on the currently
 * authenticated user's role.
 *
 * @param targetUser - The user whose permissions are being evaluated
 */
export function useUserPermissions(targetUser: User): UseUserPermissionsReturn {
  const currentUser = useAuthStore((state) => state.user);

  return useMemo(() => {
    if (!currentUser) {
      return { canEdit: false, canDelete: false, canViewAdmin: false };
    }

    const isAdmin = currentUser.role === 'admin';
    const isSelf = currentUser.id === targetUser.id;

    return {
      canEdit: isAdmin || isSelf,
      canDelete: isAdmin,
      canViewAdmin: isAdmin,
    };
  }, [currentUser, targetUser.id]);
}
```

Hook test:
```ts
// src/features/users/hooks/useUserPermissions.test.ts

import { renderHook } from '@testing-library/react';
import { describe, test, expect } from 'vitest';

import { useUserPermissions } from './useUserPermissions';

const adminUser = { id: '1', role: 'admin' as const, name: 'Admin' };
const regularUser = { id: '2', role: 'user' as const, name: 'User' };

// Mock the auth store
vi.mock('@/stores/authStore', () => ({
  useAuthStore: vi.fn().mockReturnValue(adminUser),
}));

describe('useUserPermissions', () => {
  test('admin can edit and delete any user', () => {
    const { result } = renderHook(() => useUserPermissions(regularUser));
    expect(result.current.canEdit).toBe(true);
    expect(result.current.canDelete).toBe(true);
  });
});
```

---

## TanStack Query Patterns

### Query file structure

```ts
// src/features/users/api/queries.ts

import { useQuery, queryOptions } from '@tanstack/react-query';

import { api } from '@/lib/axios';
import type { User } from '../types/user.types';

// ✅ queryOptions factory — reusable in loaders and components
export const userQueryOptions = (userId: string) =>
  queryOptions({
    queryKey: ['users', userId],
    queryFn: () => api.get<User>(`/users/${userId}`).then(r => r.data),
    staleTime: 5 * 60 * 1000,
  });

export const usersListQueryOptions = (filters?: UsersFilter) =>
  queryOptions({
    queryKey: ['users', 'list', filters],
    queryFn: () => api.get<User[]>('/users', { params: filters }).then(r => r.data),
    staleTime: 2 * 60 * 1000,
  });

// ✅ Hook wrappers for use in components
export function useUser(userId: string) {
  return useQuery(userQueryOptions(userId));
}

export function useUsersList(filters?: UsersFilter) {
  return useQuery(usersListQueryOptions(filters));
}
```

### Mutation with optimistic update

```ts
// src/features/users/api/mutations.ts

import { useMutation, useQueryClient } from '@tanstack/react-query';

import { api } from '@/lib/axios';
import type { User, UserUpdate } from '../types/user.types';

export function useUpdateUser(userId: string) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: UserUpdate) =>
      api.patch<User>(`/users/${userId}`, data).then(r => r.data),

    // Optimistic update
    onMutate: async (newData) => {
      await queryClient.cancelQueries({ queryKey: ['users', userId] });
      const previous = queryClient.getQueryData<User>(['users', userId]);
      queryClient.setQueryData<User>(['users', userId], (old) => ({
        ...old!,
        ...newData,
      }));
      return { previous };
    },
    onError: (_err, _vars, context) => {
      if (context?.previous) {
        queryClient.setQueryData(['users', userId], context.previous);
      }
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['users', userId] });
      queryClient.invalidateQueries({ queryKey: ['users', 'list'] });
    },
  });
}
```

### Route prefetching (React Router v6)

```tsx
// src/app/router.tsx
import { createBrowserRouter } from 'react-router-dom';
import { queryClient } from '@/lib/queryClient';
import { userQueryOptions } from '@/features/users/api/queries';

export const router = createBrowserRouter([
  {
    path: '/users/:userId',
    element: <UserProfile />,
    loader: ({ params }) =>
      queryClient.ensureQueryData(userQueryOptions(params.userId!)),
  },
]);
```

---

## Zustand Store Template

```ts
// src/stores/authStore.ts

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

interface AuthUser {
  id: string;
  name: string;
  email: string;
  role: 'admin' | 'user';
}

interface AuthStore {
  user: AuthUser | null;
  token: string | null;
  setAuth: (user: AuthUser, token: string) => void;
  clearAuth: () => void;
}

export const useAuthStore = create<AuthStore>()(
  devtools(
    persist(
      (set) => ({
        user: null,
        token: null,
        setAuth: (user, token) => set({ user, token }),
        clearAuth: () => set({ user: null, token: null }),
      }),
      { name: 'auth-storage' }
    ),
    { name: 'AuthStore' }
  )
);

// ✅ Always use selectors — prevents unnecessary re-renders
export const useCurrentUser = () => useAuthStore((state) => state.user);
export const useIsAuthenticated = () => useAuthStore((state) => !!state.token);
```

---

## Zod + React Hook Form Full Template

```tsx
// src/features/auth/components/RegisterForm.tsx

import { useForm, Controller } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';

import { TextField } from '@/components/ui/TextField';
import { useRegisterUser } from '../api/mutations';

// --- Schema ---

const registerSchema = z
  .object({
    name: z.string().min(2, 'Name must be at least 2 characters'),
    email: z.string().email('Invalid email address'),
    password: z.string().min(8, 'Password must be at least 8 characters'),
    confirmPassword: z.string(),
    role: z.enum(['admin', 'user']),
    acceptTerms: z.literal(true, {
      errorMap: () => ({ message: 'You must accept the terms' }),
    }),
  })
  .refine((data) => data.password === data.confirmPassword, {
    message: 'Passwords do not match',
    path: ['confirmPassword'],
  });

type RegisterFormData = z.infer<typeof registerSchema>;

// --- Component ---

export function RegisterForm() {
  const {
    register,
    control,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<RegisterFormData>({
    resolver: zodResolver(registerSchema),
    mode: 'onBlur',
    defaultValues: { role: 'user', acceptTerms: false as unknown as true },
  });

  const { mutateAsync: registerUser } = useRegisterUser();

  const onSubmit = async (data: RegisterFormData) => {
    await registerUser(data);
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)} noValidate>
      <label htmlFor="name">Full name</label>
      <input id="name" {...register('name')} />
      {errors.name && <span role="alert">{errors.name.message}</span>}

      <label htmlFor="email">Email address</label>
      <input id="email" type="email" {...register('email')} />
      {errors.email && <span role="alert">{errors.email.message}</span>}

      <label htmlFor="password">Password</label>
      <input id="password" type="password" {...register('password')} />
      {errors.password && <span role="alert">{errors.password.message}</span>}

      {/* Controller for UI library inputs */}
      <Controller
        name="role"
        control={control}
        render={({ field }) => (
          <TextField label="Role" select {...field} error={errors.role?.message} />
        )}
      />

      <label>
        <input type="checkbox" {...register('acceptTerms')} />
        I accept the terms and conditions
      </label>
      {errors.acceptTerms && <span role="alert">{errors.acceptTerms.message}</span>}

      <button type="submit" disabled={isSubmitting}>
        {isSubmitting ? 'Creating account...' : 'Create account'}
      </button>
    </form>
  );
}
```

---

## Error Boundary Implementation

```tsx
// src/components/ErrorBoundary.tsx

import { ErrorBoundary as ReactErrorBoundary, FallbackProps } from 'react-error-boundary';
import { useEffect } from 'react';

// --- Fallback components ---

export function PageErrorFallback({ error, resetErrorBoundary }: FallbackProps) {
  useEffect(() => {
    // Report to Sentry or error tracking
    console.error('Page error:', error);
  }, [error]);

  return (
    <main role="alert" style={{ padding: '2rem', textAlign: 'center' }}>
      <h1>Something went wrong</h1>
      <p>We've been notified. Please try again.</p>
      <button onClick={resetErrorBoundary}>Try again</button>
    </main>
  );
}

export function SectionErrorFallback({ resetErrorBoundary }: FallbackProps) {
  return (
    <div role="alert" style={{ padding: '1rem', border: '1px solid #e00' }}>
      <p>This section couldn't load.</p>
      <button onClick={resetErrorBoundary}>Retry</button>
    </div>
  );
}

// --- Re-export for convenience ---
export { ReactErrorBoundary as ErrorBoundary };

// --- Usage ---
/*
function ProductPage() {
  return (
    <ErrorBoundary FallbackComponent={PageErrorFallback}>
      <Header />
      <ErrorBoundary FallbackComponent={SectionErrorFallback}>
        <ProductDetails />
      </ErrorBoundary>
      <ErrorBoundary FallbackComponent={SectionErrorFallback}>
        <Reviews />
      </ErrorBoundary>
    </ErrorBoundary>
  );
}
*/
```

---

## Testing Utilities

```tsx
// src/testing/test-utils.tsx

import { ReactElement } from 'react';
import { render, RenderOptions } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';

// Fresh QueryClient per test — no shared cache between tests
function createTestQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: { retry: false, staleTime: 0 },
      mutations: { retry: false },
    },
  });
}

function AllProviders({ children }: { children: React.ReactNode }) {
  const queryClient = createTestQueryClient();
  return (
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        {children}
      </MemoryRouter>
    </QueryClientProvider>
  );
}

const customRender = (ui: ReactElement, options?: Omit<RenderOptions, 'wrapper'>) =>
  render(ui, { wrapper: AllProviders, ...options });

// Re-export everything from testing library
export * from '@testing-library/react';
export { customRender as render };
```

Test example using utilities:
```tsx
// src/features/users/components/UserCard.test.tsx

import { render, screen } from '@/testing/test-utils';
import userEvent from '@testing-library/user-event';
import { describe, test, expect, vi } from 'vitest';

import { UserCard } from './UserCard';

const user = { name: 'Jane Doe', email: 'jane@example.com' };

describe('UserCard', () => {
  test('renders user name and email', () => {
    render(<UserCard {...user} />);
    expect(screen.getByText('Jane Doe')).toBeInTheDocument();
    expect(screen.getByText('jane@example.com')).toBeInTheDocument();
  });

  test('calls onEdit with user name when Edit is clicked', async () => {
    const onEdit = vi.fn();
    render(<UserCard {...user} onEdit={onEdit} />);

    await userEvent.click(screen.getByRole('button', { name: /edit/i }));

    expect(onEdit).toHaveBeenCalledWith('Jane Doe');
  });

  test('does not render Edit button when onEdit is not provided', () => {
    render(<UserCard {...user} />);
    expect(screen.queryByRole('button', { name: /edit/i })).not.toBeInTheDocument();
  });

  test('shows inactive badge when isActive is false', () => {
    render(<UserCard {...user} isActive={false} />);
    expect(screen.getByText('Inactive')).toBeInTheDocument();
  });
});
```

---

## Vite Config with Path Aliases

```ts
// vite.config.ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/testing/setup.ts',
    css: true,
  },
});
```

```ts
// src/testing/setup.ts
import '@testing-library/jest-dom';
```

---

## TypeScript Config

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"]
    }
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

---

## ESLint Config (Flat Config)

```js
// eslint.config.js
import js from '@eslint/js';
import globals from 'globals';
import reactHooks from 'eslint-plugin-react-hooks';
import reactRefresh from 'eslint-plugin-react-refresh';
import jsxA11y from 'eslint-plugin-jsx-a11y';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  { ignores: ['dist'] },
  {
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    files: ['**/*.{ts,tsx}'],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
    },
    plugins: {
      'react-hooks': reactHooks,
      'react-refresh': reactRefresh,
      'jsx-a11y': jsxA11y,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      ...jsxA11y.configs.recommended.rules,
      'react-refresh/only-export-components': ['warn', { allowConstantExport: true }],
      '@typescript-eslint/no-explicit-any': 'error',
      '@typescript-eslint/consistent-type-imports': 'warn',
    },
  },
);
```

---

## Environment Config

```ts
// src/config/config.ts

function requireEnv(key: string): string {
  const value = import.meta.env[key];
  if (!value) {
    throw new Error(`Missing required environment variable: ${key}`);
  }
  return value;
}

const config = {
  apiBaseUrl: requireEnv('VITE_API_BASE_URL'),
  environment: import.meta.env.MODE as 'development' | 'production' | 'test',
  sentryDsn: import.meta.env.VITE_SENTRY_DSN as string | undefined,
} as const;

export default config;
```

`.env.example`:
```bash
# API
VITE_API_BASE_URL=http://localhost:8000

# Error tracking (optional in development)
VITE_SENTRY_DSN=
```

---

## TanStack Query Client Setup

```ts
// src/lib/queryClient.ts

import { QueryClient } from '@tanstack/react-query';

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 60 * 1000,         // 1 minute
      retry: 1,
      refetchOnWindowFocus: false,  // Disable if backend is slow
    },
    mutations: {
      retry: 0,
    },
  },
});
```

```tsx
// src/app/providers.tsx

import { QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';

import { queryClient } from '@/lib/queryClient';

export function AppProviders({ children }: { children: React.ReactNode }) {
  return (
    <QueryClientProvider client={queryClient}>
      {children}
      {import.meta.env.DEV && <ReactQueryDevtools />}
    </QueryClientProvider>
  );
}
```

---

## Additional Resources

- [Bulletproof React](https://github.com/alan2207/bulletproof-react) — Production React architecture reference
- [Airbnb React/JSX Style Guide](https://github.com/airbnb/javascript/tree/master/react)
- [React Official Documentation](https://react.dev)
- [React TypeScript Cheatsheet](https://react-typescript-cheatsheet.netlify.app/)
- [TanStack Query Documentation](https://tanstack.com/query/latest)
- [React Hook Form Documentation](https://react-hook-form.com/)
- [Zod Documentation](https://zod.dev/)
- [Zustand Documentation](https://docs.pmnd.rs/zustand/)
- [React Testing Library](https://testing-library.com/docs/react-testing-library/intro/)
- [Vitest Documentation](https://vitest.dev/)
- [eslint-plugin-jsx-a11y](https://github.com/jsx-eslint/eslint-plugin-jsx-a11y)
- [react-error-boundary](https://github.com/bvaughn/react-error-boundary)
