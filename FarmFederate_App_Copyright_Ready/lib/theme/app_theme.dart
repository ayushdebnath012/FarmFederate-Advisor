import 'package:flutter/material.dart';

class AppTheme {
  static const Color primaryGreen = Color(0xFF4CAF50);
  static const Color primaryGreenLight = Color(0xFF81C784);
  static const Color darkGreen = Color(0xFF2E7D32);

  static const Color backgroundDark = Color(0xFF121212);
  static const Color cardDark = Color(0xFF1E1E1E);
  static const Color surfaceDark = Color(0xFF2C2C2C);

  static const Color accentCyan = Color(0xFF00BCD4);
  static const Color accentBlue = Color(0xFF2196F3);
  static const Color accentOrange = Color(0xFFFF9800);

  static const Color healthyColor = Color(0xFF4CAF50);
  static const Color droughtColor = Color(0xFFFF9800);
  static const Color nutrientColor = Color(0xFFFFEB3B);
  static const Color diseaseColor = Color(0xFFF44336);
  static const Color pestColor = Color(0xFF9C27B0);

  static const LinearGradient primaryGradient = LinearGradient(
    colors: [primaryGreen, darkGreen],
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
  );

  static ThemeData get darkTheme => ThemeData(
        brightness: Brightness.dark,
        primaryColor: primaryGreen,
        scaffoldBackgroundColor: backgroundDark,
        colorScheme: const ColorScheme.dark(
          primary: primaryGreen,
          secondary: accentCyan,
          surface: cardDark,
        ),
        appBarTheme: const AppBarTheme(
          backgroundColor: cardDark,
          foregroundColor: Colors.white,
          elevation: 0,
        ),
        cardTheme: CardThemeData(
          color: cardDark,
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        ),
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          fillColor: surfaceDark,
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: BorderSide.none,
          ),
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: BorderSide.none,
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: const BorderSide(color: primaryGreen),
          ),
          hintStyle: const TextStyle(color: Colors.white38),
          labelStyle: const TextStyle(color: Colors.white70),
          prefixIconColor: Colors.white54,
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: primaryGreen,
            foregroundColor: Colors.white,
            shape:
                RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          ),
        ),
        outlinedButtonTheme: OutlinedButtonThemeData(
          style: OutlinedButton.styleFrom(
            foregroundColor: primaryGreen,
            side: const BorderSide(color: primaryGreen),
            shape:
                RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          ),
        ),
      );
}
