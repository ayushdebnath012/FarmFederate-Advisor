import 'package:flutter/material.dart';

class ProbBar extends StatelessWidget {
  final double value;
  final String label;
  const ProbBar({super.key, required this.value, required this.label});
  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label),
        const SizedBox(height: 4),
        LinearProgressIndicator(value: value),
        const SizedBox(height: 8),
      ],
    );
  }
}
