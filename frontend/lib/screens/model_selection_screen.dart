import 'package:flutter/material.dart';
import '../theme/app_theme.dart';
import '../constants.dart';

/// Model Selection Screen
/// Shows the 200 model configurations (5 LLM x 5 ViT x 8 VLM)
class ModelSelectionScreen extends StatefulWidget {
  final String apiBase;

  const ModelSelectionScreen({super.key, required this.apiBase});

  @override
  State<ModelSelectionScreen> createState() => _ModelSelectionScreenState();
}

class _ModelSelectionScreenState extends State<ModelSelectionScreen> {
  String _selectedLlm = BEST_LLM;
  String _selectedVit = BEST_VIT;
  String _selectedFusion = BEST_FUSION;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.backgroundDark,
      appBar: AppBar(
        title: const Text('Analysis Type'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            _buildInfoCard(),
            const SizedBox(height: 16),
            _buildSectionTitle('Text Analysis Engine', Icons.text_fields),
            const SizedBox(height: 8),
            _buildChipSelector(LLM_ENCODERS, _selectedLlm, (v) {
              setState(() => _selectedLlm = v);
            }),
            const SizedBox(height: 16),
            _buildSectionTitle('Image Analysis Engine', Icons.image),
            const SizedBox(height: 8),
            _buildChipSelector(VIT_ENCODERS, _selectedVit, (v) {
              setState(() => _selectedVit = v);
            }),
            const SizedBox(height: 16),
            _buildSectionTitle('Fusion Strategy', Icons.merge_type),
            const SizedBox(height: 8),
            _buildChipSelector(VLM_FUSIONS, _selectedFusion, (v) {
              setState(() => _selectedFusion = v);
            }),
            const SizedBox(height: 24),
            _buildCurrentSelection(),
          ],
        ),
      ),
    );
  }

  Widget _buildInfoCard() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppTheme.accentBlue.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppTheme.accentBlue.withValues(alpha: 0.3)),
      ),
      child: const Row(
        children: [
          Icon(Icons.info_outline, color: AppTheme.accentBlue),
          SizedBox(width: 12),
          Expanded(
            child: Text(
              'Choose how your crops are analyzed. The recommended settings work best for most farmers.',
              style: TextStyle(color: Colors.white70),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSectionTitle(String title, IconData icon) {
    return Row(
      children: [
        Icon(icon, color: AppTheme.primaryGreen, size: 20),
        const SizedBox(width: 8),
        Text(
          title,
          style: const TextStyle(
            color: Colors.white,
            fontWeight: FontWeight.bold,
            fontSize: 16,
          ),
        ),
      ],
    );
  }

  Widget _buildChipSelector(
      List<String> options, String selected, ValueChanged<String> onSelected) {
    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: options.map((option) {
        final isSelected = option == selected;
        final isBest = option == BEST_LLM ||
            option == BEST_VIT ||
            option == BEST_FUSION;
        return ChoiceChip(
          label: Text(
            isBest ? '$option (Best)' : option,
            style: TextStyle(
              color: isSelected ? Colors.white : Colors.white70,
              fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
            ),
          ),
          selected: isSelected,
          selectedColor: AppTheme.primaryGreen,
          backgroundColor: AppTheme.cardDark,
          onSelected: (_) => onSelected(option),
        );
      }).toList(),
    );
  }

  Widget _buildCurrentSelection() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppTheme.cardDark,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppTheme.primaryGreen.withValues(alpha: 0.3)),
      ),
      child: Column(
        children: [
          const Text(
            'Current Configuration',
            style: TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.bold,
              fontSize: 16,
            ),
          ),
          const SizedBox(height: 12),
          _buildConfigRow('Text Engine', _selectedLlm),
          _buildConfigRow('Image Engine', _selectedVit),
          _buildConfigRow('Fusion', _selectedFusion),
          const SizedBox(height: 12),
          Text(
            'Best: $BEST_LLM + $BEST_VIT + $BEST_FUSION (${(BEST_MACRO_F1_CENTRALIZED * 100).toStringAsFixed(1)}% F1)',
            style: const TextStyle(color: AppTheme.primaryGreen, fontSize: 12),
          ),
        ],
      ),
    );
  }

  Widget _buildConfigRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(color: Colors.white70)),
          Text(
            value,
            style: const TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }
}
