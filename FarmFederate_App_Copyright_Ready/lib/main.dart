import 'package:flutter/material.dart';
import 'package:firebase_core/firebase_core.dart';
import 'firebase_options.dart';
import 'routes.dart';
import 'theme/app_theme.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  try {
    await Firebase.initializeApp(
      options: DefaultFirebaseOptions.currentPlatform,
    );
  } catch (e) {
    debugPrint('Firebase initialization error: $e');
  }

  runApp(const FarmFederateApp());
}

class FarmFederateApp extends StatelessWidget {
  const FarmFederateApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'FarmFederate Advisor',
      theme: AppTheme.darkTheme,
      routes: routes,
      initialRoute: '/',
      debugShowCheckedModeBanner: false,
    );
  }
}
