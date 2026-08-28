import 'package:flutter/material.dart';
import 'constants.dart';

import 'screens/login_screen.dart';
import 'screens/register_screen.dart';
import 'screens/auth_check_screen.dart';
import 'screens/profile_screen.dart';
import 'screens/settings_screen.dart';

import 'screens/dashboard_screen.dart';
import 'screens/advisory_chat_screen.dart';
import 'screens/chat_screen.dart';

import 'screens/multimodal_diagnosis_screen.dart';
import 'screens/profile_selection_screen.dart';
import 'screens/performance_results_screen.dart';
import 'screens/farm_network_screen.dart';
import 'screens/disease_detection_screen.dart';

final Map<String, WidgetBuilder> routes = {
  '/': (context) => const AuthCheckScreen(),
  '/login': (context) => const LoginScreen(),
  '/register': (context) => const RegisterScreen(),
  '/profile': (context) => const ProfileScreen(),
  '/settings': (context) => const SettingsScreen(),
  '/dashboard': (context) => DashboardScreen(apiBase: getBackendUrl()),
  '/diagnosis': (context) =>
      MultimodalDiagnosisScreen(apiBase: getBackendUrl()),
  '/profiles': (context) => ProfileSelectionScreen(apiBase: getBackendUrl()),
  '/performance': (context) =>
      PerformanceResultsScreen(apiBase: getBackendUrl()),
  '/farm-network': (context) => FarmNetworkScreen(apiBase: getBackendUrl()),
  '/chat': (context) => AdvisoryChatScreen(apiBase: getBackendUrl()),
  '/chat-old': (context) => ChatScreen(apiBase: getBackendUrl()),
  '/disease-detect': (context) =>
      DiseaseDetectionScreen(apiBase: getBackendUrl()),
};

class AppRoutes {
  static const String home = '/';
  static const String login = '/login';
  static const String register = '/register';
  static const String profile = '/profile';
  static const String settings = '/settings';
  static const String dashboard = '/dashboard';
  static const String diagnosis = '/diagnosis';
  static const String profiles = '/profiles';
  static const String performance = '/performance';
  static const String farmNetwork = '/farm-network';
  static const String chat = '/chat';
  static const String diseaseDetect = '/disease-detect';
}
