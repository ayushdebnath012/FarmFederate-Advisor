// FarmFederate Navigation Routes
// Defines all available screens and navigation paths

import 'package:flutter/material.dart';
import 'constants.dart';

// Auth screens
import 'screens/login_screen.dart';
import 'screens/register_screen.dart';
import 'screens/auth_check_screen.dart';
import 'screens/profile_screen.dart';
import 'screens/settings_screen.dart';

// Main screens
import 'screens/dashboard_screen.dart';
import 'screens/ai_chat_screen.dart';
import 'screens/chat_screen.dart';

// Paper-aligned screens
import 'screens/multimodal_diagnosis_screen.dart';
import 'screens/model_selection_screen.dart';
import 'screens/benchmark_results_screen.dart';
import 'screens/federated_learning_screen.dart';

/// Application routes
/// Following the paper's architecture (Figure 1):
/// - Multimodal input (text + image)
/// - VLM fusion with 200 model configurations
/// - Federated learning for privacy preservation
final Map<String, WidgetBuilder> routes = {
  // Auth flow
  '/': (context) => const AuthCheckScreen(),
  '/login': (context) => const LoginScreen(),
  '/register': (context) => const RegisterScreen(),
  '/profile': (context) => const ProfileScreen(),
  '/settings': (context) => const SettingsScreen(),

  // Main dashboard with 6 tabs
  '/dashboard': (context) => DashboardScreen(apiBase: getBackendUrl()),

  // Multimodal diagnosis (paper's main use case)
  '/diagnosis': (context) => MultimodalDiagnosisScreen(apiBase: getBackendUrl()),

  // Model selection (5 LLM x 5 ViT x 8 VLM = 200 configs)
  '/models': (context) => ModelSelectionScreen(apiBase: getBackendUrl()),

  // Benchmark results (Tables 13-22, Figures 6-38)
  '/benchmarks': (context) => BenchmarkResultsScreen(apiBase: getBackendUrl()),

  // Federated learning visualization
  '/federated': (context) => FederatedLearningScreen(apiBase: getBackendUrl()),

  // AI chat interface
  '/chat': (context) => AIChatScreen(apiBase: getBackendUrl()),

  // Legacy chat (backward compatibility)
  '/chat-old': (context) => ChatScreen(apiBase: getBackendUrl()),
};

/// Route names for type-safe navigation
class AppRoutes {
  static const String home = '/';
  static const String login = '/login';
  static const String register = '/register';
  static const String profile = '/profile';
  static const String settings = '/settings';
  static const String dashboard = '/dashboard';
  static const String diagnosis = '/diagnosis';
  static const String models = '/models';
  static const String benchmarks = '/benchmarks';
  static const String federated = '/federated';
  static const String chat = '/chat';
}
