// frontend/lib/screens/auth_check_screen.dart
import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import '../constants.dart';
import '../theme/app_theme.dart';
import 'dashboard_screen.dart';
import 'login_screen.dart';

/// Authentication check screen that redirects users based on their auth status
class AuthCheckScreen extends StatefulWidget {
  const AuthCheckScreen({super.key});

  @override
  State<AuthCheckScreen> createState() => _AuthCheckScreenState();
}

class _AuthCheckScreenState extends State<AuthCheckScreen> {
  bool _initialized = false;
  bool _error = false;

  @override
  void initState() {
    super.initState();
    _checkAuth();
  }

  Future<void> _checkAuth() async {
    try {
      // Give Firebase a moment to initialize
      await Future.delayed(const Duration(milliseconds: 500));

      if (!mounted) return;

      // Try to check Firebase auth state
      try {
        final user = FirebaseAuth.instance.currentUser;
        if (mounted) {
          setState(() => _initialized = true);

          // Navigate based on auth state
          if (user != null) {
            _navigateToDashboard();
          } else {
            _navigateToLogin();
          }
        }
      } catch (e) {
        debugPrint('[AuthCheck] Firebase error: $e');
        // Firebase not available - go directly to dashboard
        if (mounted) {
          _navigateToDashboard();
        }
      }
    } catch (e) {
      debugPrint('[AuthCheck] Error: $e');
      if (mounted) {
        setState(() => _error = true);
        // On any error, go to dashboard after brief delay
        await Future.delayed(const Duration(seconds: 1));
        if (mounted) {
          _navigateToDashboard();
        }
      }
    }
  }

  void _navigateToDashboard() {
    Navigator.of(context).pushReplacement(
      MaterialPageRoute(
        builder: (context) => DashboardScreen(apiBase: getBackendUrl()),
      ),
    );
  }

  void _navigateToLogin() {
    Navigator.of(context).pushReplacement(
      MaterialPageRoute(
        builder: (context) => const LoginScreen(),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.backgroundDark,
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // FarmFederate logo/icon
            Container(
              padding: const EdgeInsets.all(24),
              decoration: const BoxDecoration(
                gradient: AppTheme.primaryGradient,
                shape: BoxShape.circle,
              ),
              child: const Icon(
                Icons.agriculture,
                size: 64,
                color: Colors.white,
              ),
            ),
            const SizedBox(height: 32),
            if (!_error) ...[
              const CircularProgressIndicator(
                valueColor: AlwaysStoppedAnimation<Color>(AppTheme.primaryGreen),
              ),
              const SizedBox(height: 20),
            ],
            const Text(
              'FarmFederate',
              style: TextStyle(
                color: AppTheme.primaryGreenLight,
                fontSize: 28,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              _error
                  ? 'Starting app...'
                  : 'Your Smart Crop Health Assistant',
              textAlign: TextAlign.center,
              style: TextStyle(
                color: Colors.white.withValues(alpha: 0.7),
                fontSize: 14,
              ),
            ),
            if (_error) ...[
              const SizedBox(height: 16),
              TextButton(
                onPressed: _navigateToDashboard,
                child: const Text('Continue to App'),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
