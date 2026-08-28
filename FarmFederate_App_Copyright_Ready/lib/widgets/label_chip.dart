import 'package:flutter/material.dart';

class LabelChip extends StatelessWidget {
  final String label;
  final double prob;
  final bool active;
  const LabelChip({
    super.key,
    required this.label,
    required this.prob,
    this.active = false,
  });
  @override
  Widget build(BuildContext context) {
    return Chip(
      backgroundColor: active ? Colors.green[100] : Colors.grey[200],
      label: Text("$label (${(prob * 100).toStringAsFixed(1)}%)"),
    );
  }
}
