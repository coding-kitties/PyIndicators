import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/PyIndicators/__docusaurus/debug',
    component: ComponentCreator('/PyIndicators/__docusaurus/debug', '4ab'),
    exact: true
  },
  {
    path: '/PyIndicators/__docusaurus/debug/config',
    component: ComponentCreator('/PyIndicators/__docusaurus/debug/config', 'b50'),
    exact: true
  },
  {
    path: '/PyIndicators/__docusaurus/debug/content',
    component: ComponentCreator('/PyIndicators/__docusaurus/debug/content', '0ce'),
    exact: true
  },
  {
    path: '/PyIndicators/__docusaurus/debug/globalData',
    component: ComponentCreator('/PyIndicators/__docusaurus/debug/globalData', '3ac'),
    exact: true
  },
  {
    path: '/PyIndicators/__docusaurus/debug/metadata',
    component: ComponentCreator('/PyIndicators/__docusaurus/debug/metadata', '307'),
    exact: true
  },
  {
    path: '/PyIndicators/__docusaurus/debug/registry',
    component: ComponentCreator('/PyIndicators/__docusaurus/debug/registry', '300'),
    exact: true
  },
  {
    path: '/PyIndicators/__docusaurus/debug/routes',
    component: ComponentCreator('/PyIndicators/__docusaurus/debug/routes', 'de3'),
    exact: true
  },
  {
    path: '/PyIndicators/',
    component: ComponentCreator('/PyIndicators/', '2c8'),
    routes: [
      {
        path: '/PyIndicators/',
        component: ComponentCreator('/PyIndicators/', 'fd0'),
        routes: [
          {
            path: '/PyIndicators/tags',
            component: ComponentCreator('/PyIndicators/tags', '26f'),
            exact: true
          },
          {
            path: '/PyIndicators/tags/lagging',
            component: ComponentCreator('/PyIndicators/tags/lagging', '0ce'),
            exact: true
          },
          {
            path: '/PyIndicators/tags/real-time',
            component: ComponentCreator('/PyIndicators/tags/real-time', 'a4e'),
            exact: true
          },
          {
            path: '/PyIndicators/',
            component: ComponentCreator('/PyIndicators/', '703'),
            routes: [
              {
                path: '/PyIndicators/indicators/helpers/crossover',
                component: ComponentCreator('/PyIndicators/indicators/helpers/crossover', '2f6'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/helpers/crossunder',
                component: ComponentCreator('/PyIndicators/indicators/helpers/crossunder', '8a5'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/helpers/has-any-lower-then-threshold',
                component: ComponentCreator('/PyIndicators/indicators/helpers/has-any-lower-then-threshold', 'b2f'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/helpers/is-crossover',
                component: ComponentCreator('/PyIndicators/indicators/helpers/is-crossover', 'c1d'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/helpers/is-crossunder',
                component: ComponentCreator('/PyIndicators/indicators/helpers/is-crossunder', '733'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/helpers/is-downtrend',
                component: ComponentCreator('/PyIndicators/indicators/helpers/is-downtrend', '906'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/helpers/is-uptrend',
                component: ComponentCreator('/PyIndicators/indicators/helpers/is-uptrend', 'a2b'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/helpers/overview',
                component: ComponentCreator('/PyIndicators/indicators/helpers/overview', 'a23'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/momentum/adx',
                component: ComponentCreator('/PyIndicators/indicators/momentum/adx', 'cb7'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/momentum/macd',
                component: ComponentCreator('/PyIndicators/indicators/momentum/macd', '3af'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/momentum/momentum-confluence',
                component: ComponentCreator('/PyIndicators/indicators/momentum/momentum-confluence', '2c4'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/momentum/momentum-cycle-sentry',
                component: ComponentCreator('/PyIndicators/indicators/momentum/momentum-cycle-sentry', '750'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/momentum/overview',
                component: ComponentCreator('/PyIndicators/indicators/momentum/overview', 'ca0'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/momentum/rsi',
                component: ComponentCreator('/PyIndicators/indicators/momentum/rsi', 'f93'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/momentum/stochastic-oscillator',
                component: ComponentCreator('/PyIndicators/indicators/momentum/stochastic-oscillator', '2f3'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/momentum/wilders-rsi',
                component: ComponentCreator('/PyIndicators/indicators/momentum/wilders-rsi', 'eac'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/momentum/williams-r',
                component: ComponentCreator('/PyIndicators/indicators/momentum/williams-r', 'd09'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/momentum/z-score-predictive-zones',
                component: ComponentCreator('/PyIndicators/indicators/momentum/z-score-predictive-zones', '094'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/pattern-recognition/bearish-divergence',
                component: ComponentCreator('/PyIndicators/indicators/pattern-recognition/bearish-divergence', '6aa'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/pattern-recognition/bullish-divergence',
                component: ComponentCreator('/PyIndicators/indicators/pattern-recognition/bullish-divergence', '638'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/pattern-recognition/detect-peaks',
                component: ComponentCreator('/PyIndicators/indicators/pattern-recognition/detect-peaks', '4f8'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/pattern-recognition/overview',
                component: ComponentCreator('/PyIndicators/indicators/pattern-recognition/overview', '738'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/support-resistance/accumulation-distribution-zones',
                component: ComponentCreator('/PyIndicators/indicators/support-resistance/accumulation-distribution-zones', '281'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/support-resistance/breaker-blocks',
                component: ComponentCreator('/PyIndicators/indicators/support-resistance/breaker-blocks', '9d3'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/support-resistance/buyside-sellside-liquidity',
                component: ComponentCreator('/PyIndicators/indicators/support-resistance/buyside-sellside-liquidity', '094'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/support-resistance/fair-value-gap',
                component: ComponentCreator('/PyIndicators/indicators/support-resistance/fair-value-gap', '9cc'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/support-resistance/fibonacci-retracement',
                component: ComponentCreator('/PyIndicators/indicators/support-resistance/fibonacci-retracement', 'a65'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/support-resistance/golden-zone',
                component: ComponentCreator('/PyIndicators/indicators/support-resistance/golden-zone', '75f'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/support-resistance/golden-zone-signal',
                component: ComponentCreator('/PyIndicators/indicators/support-resistance/golden-zone-signal', '566'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/support-resistance/internal-external-liquidity-zones',
                component: ComponentCreator('/PyIndicators/indicators/support-resistance/internal-external-liquidity-zones', 'b5a'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/support-resistance/liquidity-levels-voids',
                component: ComponentCreator('/PyIndicators/indicators/support-resistance/liquidity-levels-voids', '44f'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/support-resistance/liquidity-pools',
                component: ComponentCreator('/PyIndicators/indicators/support-resistance/liquidity-pools', 'feb'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/support-resistance/liquidity-sweeps',
                component: ComponentCreator('/PyIndicators/indicators/support-resistance/liquidity-sweeps', 'a16'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/support-resistance/market-structure-break',
                component: ComponentCreator('/PyIndicators/indicators/support-resistance/market-structure-break', '71a'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/support-resistance/market-structure-choch-bos',
                component: ComponentCreator('/PyIndicators/indicators/support-resistance/market-structure-choch-bos', 'd05'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/support-resistance/mitigation-blocks',
                component: ComponentCreator('/PyIndicators/indicators/support-resistance/mitigation-blocks', '678'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/support-resistance/opening-gap',
                component: ComponentCreator('/PyIndicators/indicators/support-resistance/opening-gap', 'a6c'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/support-resistance/optimal-trade-entry',
                component: ComponentCreator('/PyIndicators/indicators/support-resistance/optimal-trade-entry', '1f7'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/support-resistance/order-blocks',
                component: ComponentCreator('/PyIndicators/indicators/support-resistance/order-blocks', 'd71'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/support-resistance/overview',
                component: ComponentCreator('/PyIndicators/indicators/support-resistance/overview', '9a3'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/support-resistance/premium-discount-zones',
                component: ComponentCreator('/PyIndicators/indicators/support-resistance/premium-discount-zones', '319'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/support-resistance/pure-price-action-liquidity-sweeps',
                component: ComponentCreator('/PyIndicators/indicators/support-resistance/pure-price-action-liquidity-sweeps', '296'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/support-resistance/range-intelligence',
                component: ComponentCreator('/PyIndicators/indicators/support-resistance/range-intelligence', '942'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/support-resistance/rejection-blocks',
                component: ComponentCreator('/PyIndicators/indicators/support-resistance/rejection-blocks', '4e4'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/support-resistance/strong-weak-high-low',
                component: ComponentCreator('/PyIndicators/indicators/support-resistance/strong-weak-high-low', '633'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/support-resistance/trendline-breakout-navigator',
                component: ComponentCreator('/PyIndicators/indicators/support-resistance/trendline-breakout-navigator', '706'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/support-resistance/volume-imbalance',
                component: ComponentCreator('/PyIndicators/indicators/support-resistance/volume-imbalance', 'c60'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/support-resistance/volumetric-supply-demand-zones',
                component: ComponentCreator('/PyIndicators/indicators/support-resistance/volumetric-supply-demand-zones', '65a'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/trend/ema',
                component: ComponentCreator('/PyIndicators/indicators/trend/ema', '97d'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/trend/ema-trend-ribbon',
                component: ComponentCreator('/PyIndicators/indicators/trend/ema-trend-ribbon', 'a21'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/trend/overview',
                component: ComponentCreator('/PyIndicators/indicators/trend/overview', '4db'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/trend/pulse-mean-accelerator',
                component: ComponentCreator('/PyIndicators/indicators/trend/pulse-mean-accelerator', '303'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/trend/sma',
                component: ComponentCreator('/PyIndicators/indicators/trend/sma', '875'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/trend/supertrend',
                component: ComponentCreator('/PyIndicators/indicators/trend/supertrend', '8b9'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/trend/supertrend-clustering',
                component: ComponentCreator('/PyIndicators/indicators/trend/supertrend-clustering', '71e'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/trend/volume-weighted-trend',
                component: ComponentCreator('/PyIndicators/indicators/trend/volume-weighted-trend', '157'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/trend/wma',
                component: ComponentCreator('/PyIndicators/indicators/trend/wma', 'dc6'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/trend/zero-lag-ema-envelope',
                component: ComponentCreator('/PyIndicators/indicators/trend/zero-lag-ema-envelope', '3fb'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/volatility/atr',
                component: ComponentCreator('/PyIndicators/indicators/volatility/atr', 'e94'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/volatility/bollinger-bands',
                component: ComponentCreator('/PyIndicators/indicators/volatility/bollinger-bands', '930'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/volatility/bollinger-overshoot',
                component: ComponentCreator('/PyIndicators/indicators/volatility/bollinger-overshoot', '755'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/volatility/moving-average-envelope',
                component: ComponentCreator('/PyIndicators/indicators/volatility/moving-average-envelope', '6f8'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/volatility/nadaraya-watson-envelope',
                component: ComponentCreator('/PyIndicators/indicators/volatility/nadaraya-watson-envelope', 'ede'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/indicators/volatility/overview',
                component: ComponentCreator('/PyIndicators/indicators/volatility/overview', '68b'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/installation',
                component: ComponentCreator('/PyIndicators/installation', '273'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/PyIndicators/',
                component: ComponentCreator('/PyIndicators/', 'de1'),
                exact: true,
                sidebar: "docs"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
