import 'package:flutter/material.dart';

class FarmNetworkScreen extends StatefulWidget {
  final String apiBase;

  const FarmNetworkScreen({Key? key, required this.apiBase}) : super(key: key);

  @override
  State<FarmNetworkScreen> createState() => _FarmNetworkScreenState();
}

class _FarmNetworkScreenState extends State<FarmNetworkScreen> {
  bool _isSync = false;
  final int _currentRound = 3;
  final int _totalRounds = 8;
  final int _participatingFarms = 3;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0A0E21),
      appBar: AppBar(
        elevation: 0,
        backgroundColor: const Color(0xFF1D1E33),
        title: const Text('Farm Network'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildPrivacyBanner(),
            const SizedBox(height: 24),
            _buildGroupComparison(),
            const SizedBox(height: 24),
            _buildSyncStatus(),
            const SizedBox(height: 24),
            _buildParticipatingFarms(),
            const SizedBox(height: 24),
            _buildPrivacyGuarantees(),
            const SizedBox(height: 24),
            _buildSyncHistory(),
          ],
        ),
      ),
    );
  }

  Widget _buildPrivacyBanner() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          colors: [Color(0xFF134E5E), Color(0xFF71B280)],
        ),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.2),
              borderRadius: BorderRadius.circular(12),
            ),
            child: const Icon(Icons.security, color: Colors.white, size: 32),
          ),
          const SizedBox(width: 16),
          const Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Network Text | Image | Fusion',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                SizedBox(height: 4),
                Text(
                  'Three independent groups calibrated with secure weighted update - K=3 clients, T=8 rounds, E=3 local passes. Raw farm data never leaves your device.',
                  style: TextStyle(color: Colors.white70, fontSize: 12),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildGroupComparison() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFF1D1E33),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Cross-Group Results',
            style: TextStyle(
              color: Colors.white,
              fontSize: 18,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 4),
          const Text(
            'All groups calibrated under identical secure weighted update conditions',
            style: TextStyle(color: Colors.white54, fontSize: 11),
          ),
          const SizedBox(height: 16),
          _buildGroupCard(
            group: 'Network Text',
            bestProfile: 'Text Core A',
            modality: 'Text (agronomist symptom descriptions)',
            centF1: 0.427,
            fedF1: 0.460,
            retention: 107.7,
            color: const Color(0xFF4169E1),
            icon: Icons.text_fields,
            note: 'Exceeds baseline result in network setting',
          ),
          const SizedBox(height: 12),
          _buildGroupCard(
            group: 'Network Image',
            bestProfile: 'Image Core A',
            modality: 'Image (tea leaf photographs)',
            centF1: 0.886,
            fedF1: 0.886,
            retention: 100.0,
            color: const Color(0xFF228B22),
            icon: Icons.image,
            note: 'Perfect retention in this network run',
          ),
          const SizedBox(height: 12),
          _buildGroupCard(
            group: 'Network Fusion',
            bestProfile: 'Fusion Core A',
            modality: 'Text + Image combined',
            centF1: 0.873,
            fedF1: 0.861,
            retention: 98.6,
            color: const Color(0xFFB22222),
            icon: Icons.merge_type,
            note: 'Best overall network performance',
            isBest: true,
          ),
          const SizedBox(height: 16),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.05),
              borderRadius: BorderRadius.circular(8),
            ),
            child: const Row(
              children: [
                Icon(Icons.info_outline, color: Colors.white38, size: 14),
                SizedBox(width: 8),
                Expanded(
                  child: Text(
                    'Avg Net/Base retention: 102.1% across all 3 groups',
                    style: TextStyle(color: Colors.white54, fontSize: 11),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildGroupCard({
    required String group,
    required String bestProfile,
    required String modality,
    required double centF1,
    required double fedF1,
    required double retention,
    required Color color,
    required IconData icon,
    required String note,
    bool isBest = false,
  }) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: color.withValues(alpha: isBest ? 0.6 : 0.3),
          width: isBest ? 1.5 : 1,
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, color: color, size: 18),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  group,
                  style: TextStyle(
                    color: color,
                    fontWeight: FontWeight.bold,
                    fontSize: 15,
                  ),
                ),
              ),
              if (isBest)
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 7,
                    vertical: 2,
                  ),
                  decoration: BoxDecoration(
                    color: color.withValues(alpha: 0.2),
                    borderRadius: BorderRadius.circular(4),
                  ),
                  child: Text(
                    'BEST',
                    style: TextStyle(
                      color: color,
                      fontSize: 9,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            '$bestProfile | $modality',
            style: const TextStyle(color: Colors.white60, fontSize: 11),
          ),
          const SizedBox(height: 10),
          Row(
            children: [
              _buildF1Chip('Base', centF1, Colors.white54),
              const SizedBox(width: 8),
              _buildF1Chip('Net', fedF1, color),
              const SizedBox(width: 8),
              _buildRetentionChip(retention, color),
            ],
          ),
          const SizedBox(height: 6),
          Text(
            note,
            style: const TextStyle(color: Colors.white38, fontSize: 10),
          ),
        ],
      ),
    );
  }

  Widget _buildF1Chip(String label, double value, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.15),
        borderRadius: BorderRadius.circular(6),
      ),
      child: Text(
        '$label F1 ${value.toStringAsFixed(3)}',
        style: TextStyle(
          color: color,
          fontSize: 11,
          fontWeight: FontWeight.w600,
        ),
      ),
    );
  }

  Widget _buildRetentionChip(double retention, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.05),
        borderRadius: BorderRadius.circular(6),
      ),
      child: Text(
        '${retention.toStringAsFixed(1)}% retained',
        style: const TextStyle(color: Colors.white54, fontSize: 11),
      ),
    );
  }

  Widget _buildSyncStatus() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFF1D1E33),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                'Sync Status',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 12,
                  vertical: 6,
                ),
                decoration: BoxDecoration(
                  color: _isSync
                      ? Colors.green.withValues(alpha: 0.2)
                      : Colors.grey.withValues(alpha: 0.2),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(
                    color: _isSync ? Colors.green : Colors.grey,
                  ),
                ),
                child: Text(
                  _isSync ? 'Sync' : 'Idle',
                  style: TextStyle(
                    color: _isSync ? Colors.green : Colors.grey,
                    fontSize: 12,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 20),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                'Round $_currentRound of $_totalRounds  |  $_participatingFarms clients',
                style: const TextStyle(color: Colors.white70, fontSize: 14),
              ),
              Text(
                '${((_currentRound / _totalRounds) * 100).toStringAsFixed(0)}%',
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 14,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          LinearProgressIndicator(
            value: _currentRound / _totalRounds,
            backgroundColor: Colors.white12,
            valueColor: const AlwaysStoppedAnimation(Color(0xFF1D976C)),
          ),
          const SizedBox(height: 20),
          ElevatedButton.icon(
            onPressed: () => setState(() => _isSync = !_isSync),
            icon: Icon(_isSync ? Icons.stop : Icons.play_arrow),
            label: Text(_isSync ? 'Stop Sync' : 'Start Sync'),
            style: ElevatedButton.styleFrom(
              backgroundColor: _isSync ? Colors.red : const Color(0xFF1D976C),
              minimumSize: const Size(double.infinity, 48),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildParticipatingFarms() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFF1D1E33),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                'Participating Farms',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 12,
                  vertical: 6,
                ),
                decoration: BoxDecoration(
                  color: Colors.blue.withValues(alpha: 0.2),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  '$_participatingFarms clients (K=3)',
                  style: const TextStyle(
                    color: Colors.blue,
                    fontSize: 12,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          _buildFarmCard(
            'Punjab Farm (Client 1)',
            'Online',
            'Contributing',
            Colors.green,
          ),
          const SizedBox(height: 8),
          _buildFarmCard(
            'Kerala Farm (Client 2)',
            'Online',
            'Sync',
            Colors.blue,
          ),
          const SizedBox(height: 8),
          _buildFarmCard('UP Farm (Client 3)', 'Online', 'Idle', Colors.orange),
        ],
      ),
    );
  }

  Widget _buildFarmCard(
    String name,
    String status,
    String activity,
    Color color,
  ) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.3),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color.withValues(alpha: 0.3)),
      ),
      child: Row(
        children: [
          Container(
            width: 10,
            height: 10,
            decoration: BoxDecoration(color: color, shape: BoxShape.circle),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  name,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                Text(
                  activity,
                  style: const TextStyle(color: Colors.white60, fontSize: 11),
                ),
              ],
            ),
          ),
          Text(
            status,
            style: TextStyle(
              color: color,
              fontSize: 12,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildPrivacyGuarantees() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFF1D1E33),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.shield, color: Colors.green, size: 24),
              SizedBox(width: 12),
              Text(
                'Privacy Guarantees',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          _buildGuaranteeItem(
            Icons.lock_outline,
            'End-to-End Encryption',
            'All communications are encrypted',
          ),
          const SizedBox(height: 12),
          _buildGuaranteeItem(
            Icons.visibility_off,
            'No Raw Data Sharing',
            'Only summary updates are transmitted via secure weighted update',
          ),
          const SizedBox(height: 12),
          _buildGuaranteeItem(
            Icons.group,
            'Non-IID Dirichlet Splits',
            'Realistic heterogeneous data distribution (alpha=1.0)',
          ),
          const SizedBox(height: 12),
          _buildGuaranteeItem(
            Icons.verified_user,
            'Secure Aggregation',
            'Central server cannot see individual client updates',
          ),
        ],
      ),
    );
  }

  Widget _buildGuaranteeItem(IconData icon, String title, String description) {
    return Row(
      children: [
        Container(
          padding: const EdgeInsets.all(8),
          decoration: BoxDecoration(
            color: Colors.green.withValues(alpha: 0.2),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Icon(icon, color: Colors.green, size: 20),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                ),
              ),
              Text(
                description,
                style: const TextStyle(color: Colors.white60, fontSize: 11),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildSyncHistory() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFF1D1E33),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Sync History',
            style: TextStyle(
              color: Colors.white,
              fontSize: 18,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 16),
          _buildHistoryItem('Round 8', '2 hours ago', 'Fusion F1: 0.861'),
          const SizedBox(height: 8),
          _buildHistoryItem('Round 5', '1 day ago', 'Fusion F1: 0.849'),
          const SizedBox(height: 8),
          _buildHistoryItem('Round 1', '2 days ago', 'Fusion F1: 0.821'),
        ],
      ),
    );
  }

  Widget _buildHistoryItem(String round, String time, String metric) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.3),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        children: [
          const Icon(Icons.check_circle, color: Colors.green, size: 20),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  round,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                Text(
                  time,
                  style: const TextStyle(color: Colors.white60, fontSize: 11),
                ),
              ],
            ),
          ),
          Text(
            metric,
            style: const TextStyle(
              color: Colors.green,
              fontSize: 13,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }
}
