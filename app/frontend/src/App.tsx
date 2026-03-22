import { useAuth } from '@/contexts/auth-context';
import { Layout } from './components/Layout';
import { LoginPage } from './components/LoginPage';
import { Toaster } from './components/ui/sonner';

export default function App() {
  const { isAuthenticated } = useAuth();

  return (
    <>
      {isAuthenticated ? <Layout /> : <LoginPage />}
      <Toaster />
    </>
  );
}
