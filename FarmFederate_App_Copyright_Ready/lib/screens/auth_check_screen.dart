import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import '../constants.dart';
import '../theme/app_theme.dart';
import 'dashboard_screen.dart';
import 'login_screen.dart';

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
      await Future.delayed(const Duration(milliseconds: 500));

      if (!mounted) return;

      try {
        final user = FirebaseAuth.instance.currentUser;
        if (mounted) {
          setState(() => _initialized = true);

          if (user != null) {
            _navigateToDashboard();
          } else {
            _navigateToLogin();
          }
        }
      } catch (e) {
        debugPrint('[AuthCheck] Firebase error: $e');
        if (mounted) {
          _navigateToDashboard();
        }
      }
    } catch (e) {
      debugPrint('[AuthCheck] Error: $e');
      if (mounted) {
        setState(() => _error = true);
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
      MaterialPageRoute(builder: (context) => const LoginScreen()),
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
            ClipRRect(
              borderRadius: BorderRadius.circular(20),
              child: Image.asset('assets/logo.png', width: 112, height: 112),
            ),
            const SizedBox(height: 32),
            if (!_error) ...[
              const CircularProgressIndicator(
                valueColor: AlwaysStoppedAnimation<Color>(
                  AppTheme.primaryGreen,
                ),
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
              _error ? 'Starting app...' : 'Your Tea Disease Assistant',
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
